import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.optimizers import L_BFGS_B

# Set Roman font and bold ticks globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # You can also try 'DejaVu Serif'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12# If your code expects a coupling_map, set it explicitly or leave as None
coupling_map = None


plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
# Setup dataset paths and parameters
base_path = "/home/ashok/Documents/battery_alt_dataset/battery_alt_dataset/regular_alt_batteries"
battery_map = {
    "9.3A": ["battery01.csv", "battery11.csv"],
    "12.9A": ["battery31.csv", "battery22.csv"],
    "14.3A": ["battery23.csv", "battery52.csv"],
    "16.0A": ["battery00.csv", "battery10.csv"],
    "19.0A": ["battery20.csv", "battery30.csv"]
}
color_map = {
    "9.3A": "tab:blue",
    "12.9A": "mediumorchid",
    "14.3A": "limegreen",
    "16.0A": "darkorange",
    "19.0A": "red"
}
NUM_POINTS = 30000
CUTOFF_VOLTAGE = 5# If your code expects a coupling_map, set it explicitly or leave as None
coupling_map = None


plt.figure(figsize=(10, 6))

for current, files in battery_map.items():
    interpolated_voltages = []
    min_cycle_duration = np.inf
    # Process each file for the current battery type
    for file in files:
        df = pd.read_csv(os.path.join(base_path, file))
        df_discharge = df[(df["mode"] == -1) &
                          (df["voltage_load"] > 5) &
                          (df["voltage_load"] < 8.4)].copy()
        df_discharge["cycle"] = (df_discharge["time"].diff() > 10).cumsum()
        # Process first 4 cycles
        for i in range(4):  # First 4 cycles
            cycle = df_discharge[df_discharge["cycle"] == i].copy()
            if not cycle.empty and len(cycle) > 10:
                cycle["time"] = cycle["time"] - cycle["time"].min()
                cycle = cycle.sort_values("time")
                duration = cycle["time"].max()
                min_cycle_duration = min(min_cycle_duration, duration)

                try:
                    interp_func = interp1d(cycle["time"], cycle["voltage_load"], kind='linear',
                                           bounds_error=False, fill_value="extrapolate")
                    interpolated_voltages.append(interp_func)
                except Exception as e:
                    print(f"Interpolation failed for {file}, cycle {i}: {e}")
    # If we have any interpolated voltages, proceed with averaging and QNN
    if interpolated_voltages:
        common_time = np.linspace(0, min_cycle_duration, NUM_POINTS)
        interpolated_arrays = [f(common_time) for f in interpolated_voltages]
        avg_voltage = np.mean(interpolated_arrays, axis=0)
        avg_voltage_clipped = np.clip(avg_voltage, CUTOFF_VOLTAGE, None)
        smoothed_voltage = uniform_filter1d(avg_voltage_clipped, size=20)

        # features and target
        X = common_time.reshape(-1, 1)  # Feature matrix
        y = avg_voltage.reshape(-1, 1)  # Target vector
        # Scale features and target
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        # Improved QNN: 1 input, 1 weight parameter, single Ry for each
        param_x = Parameter("x")
        param_y = Parameter("y")
        feature_map = QuantumCircuit(1)
        feature_map.ry(param_x, 0)
        ansatz = QuantumCircuit(1)
        ansatz.ry(param_y, 0)

        qc = QuantumCircuit(1)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        estimator = Estimator()
        qnn = EstimatorQNN(
        # Predict *over the entire time range* for smooth curve
            circuit=qc,
            input_params=[param_x],
            weight_params=[param_y],
            estimator=estimator,
        )

        regressor = NeuralNetworkRegressor(
            neural_network=qnn,
            optimizer=L_BFGS_B(maxiter=500),
        )

        # Train QNN
        regressor.fit(X_train, y_train.ravel())

        # 2. Predict for TRAIN set (CRITICAL: Move this here to define y_true_train)
        y_pred_scaled_train = regressor.predict(X_train)
        y_pred_train = scaler_y.inverse_transform(np.array(y_pred_scaled_train).reshape(-1, 1))
        y_true_train = scaler_y.inverse_transform(y_train.reshape(-1, 1))

        # 3. Predict for TEST set
        y_pred_scaled_test = regressor.predict(X_test)
        y_pred_test = scaler_y.inverse_transform(np.array(y_pred_scaled_test).reshape(-1, 1))
        y_true_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        X_test_inv = scaler_X.inverse_transform(X_test)

        # 4. ===== Confidence Interval (CI) Calculation =====
        # Now y_true_train and y_pred_train are defined
        residuals = y_true_train.flatten() - y_pred_train.flatten()
        sigma = np.std(residuals)
        
        # 95% Prediction Interval bounds
        upper_ci = y_pred_test.flatten() + (1.96 * sigma)
        lower_ci = y_pred_test.flatten() - (1.96 * sigma)

        # 5. Metrics Calculation (Train & Test)
        mae_train = mean_absolute_error(y_true_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
        r2_train = r2_score(y_true_train, y_pred_train)

        mae_test = mean_absolute_error(y_true_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
        r2_test = r2_score(y_true_test, y_pred_test)

        # ... (Rest of your printing and plotting code remains the same)


        print(f"\nPerformance Metrics for {current} (Train):")
        print(f"  MAE: {mae_train:.4f}")
        print(f"  RMSE: {rmse_train:.4f}")
        print(f"  R²: {r2_train:.4f}")

        print(f"\nPerformance Metrics for {current} (Test):")
        print(f"  MAE: {mae_test:.4f}")
        print(f"  RMSE: {rmse_test:.4f}")
        print(f"  R²: {r2_test:.4f}")

        # Plot true and predicted curves
        plt.plot(common_time, smoothed_voltage, color=color_map[current], label=f"{current} ")

        # Smooth predicted curve for better comparison (test set)
        sorted_idx = np.argsort(X_test_inv.ravel())
        X_sorted = X_test_inv[sorted_idx]
        y_pred_sorted = y_pred_test[sorted_idx]
        plt.plot(X_sorted,  y_pred_sorted , linestyle='--', color=color_map[current])
        # Prepare CI for plotting
        # Print counts
        print(f"Number of training points for {current}: {len(X_train)}")
        print(f"Number of test points for {current}: {len(X_test)}")
         # ===== Sorting for Plot =====
        idx = np.argsort(X_test_inv.flatten())
        t_sorted = X_test_inv.flatten()[idx]
        y_sorted = y_pred_test.flatten()[idx]
        upper_sorted = upper_ci[idx]
        lower_sorted = lower_ci[idx]

        plt.fill_between(
        t_sorted,
        lower_sorted,
        upper_sorted,
        color=color_map[current],
        alpha=0.25,
        
    )

# Labels and title
plt.xlabel("Time [s]", fontweight='bold', fontsize=13, fontname='Times New Roman')
plt.ylabel("Voltage [V](2S)", fontweight='bold', fontsize=13, fontname='Times New Roman')

# Tick labels bold + Roman
ax = plt.gca()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Times New Roman')

from matplotlib.font_manager import FontProperties
roman_bold = FontProperties(family='serif', weight='bold', size=12)

plt.legend(prop=roman_bold)
plt.grid(True)
plt.tight_layout()
plt.show()
