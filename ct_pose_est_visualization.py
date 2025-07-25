# CT Pose Estimation Model Visualization & Plotting
# created 07/20/2025

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from ct_pose_estimation_model import PoseEstRegression
from torch.utils.data import DataLoader
from ct_sweep_dataset import CTSweepDataset, parse_dataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

matplotlib.rcParams['font.family'] = 'Arial'

# load excel file
pitch_df = pd.read_excel("/Users/emiz/Desktop/pose_est_model_data.xlsx", sheet_name="Roll w Validation", engine='openpyxl', skiprows=2)
# roll_df = pd.read_excel("/Users/emiz/Desktop/pose_est_model_data.xlsx", sheet_name="Roll w Validation", engine='openpyxl', skiprows=2)

epochs = np.arange(1, 51)

# Manually extract the MAE columns for each run
train_mae1 = pitch_df['MAE (°)']
train_mae2 = pitch_df['MAE (°).2']
train_mae3 = pitch_df['MAE (°).4']

test_mae1 = pitch_df['MAE (°).1']
test_mae2 = pitch_df['MAE (°).3']
test_mae3 = pitch_df['MAE (°).5']

# Combine for stats
train_mae_stack = np.vstack([train_mae1, train_mae2, train_mae3]).T
train_mae_mean = np.mean(train_mae_stack, axis=1)
train_mae_std = np.std(train_mae_stack, axis=1)

test_mae_stack = np.vstack([test_mae1, test_mae2, test_mae3]).T
test_mae_mean = np.mean(test_mae_stack, axis=1)
test_mae_std = np.std(test_mae_stack, axis=1)

# Plot
plt.figure(figsize=(10, 6))

plt.plot(epochs, train_mae_mean, label="Train MAE", color="green", linewidth=2)
plt.fill_between(epochs, train_mae_mean - train_mae_std, train_mae_mean + train_mae_std, alpha=0.25, color="green", label="±1 Std Dev")

plt.plot(epochs, test_mae_mean, label="Test MAE", color="blue", linewidth=2)
plt.fill_between(epochs, test_mae_mean - test_mae_std, test_mae_mean + test_mae_std, alpha=0.25, color="blue", label="±1 Std Dev")

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MAE (°)", fontsize=12)
plt.title("Roll Model Learning Curve (Smooth L1 Loss ± Std)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

########## BLAND ALTMAN PLOT ##########

def bland_altman(model_path, test_path, label_type):
    
    # Load model
    model = PoseEstRegression()  # Update this if needed
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Load your test data
    test_imgs, test_labels = parse_dataset(test_path, label_type)
    test_dataset = CTSweepDataset(test_imgs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for img, label in test_loader:
            output = model(img)
            y_pred.append(output.item())
            y_true.append(label.item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    means = (y_true + y_pred) / 2
    diffs = y_pred - y_true
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    plt.figure(figsize=(8, 5))
    plt.scatter(means, diffs, alpha=0.6)
    plt.axhline(mean_diff, color='red', linestyle='--', label=f'Mean bias = {mean_diff:.2f}°')
    plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--', label='+1.96 SD')
    plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--', label='–1.96 SD')
    plt.xlabel('Mean of Prediction and Ground Truth (°)')
    plt.ylabel('Prediction – Ground Truth (°)')
    plt.title('Bland–Altman Plot for Pitch Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def correlation_plot(model_path, test_path, label_type):

    # load model
    model = PoseEstRegression()  # Update this if needed
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # load test data
    test_imgs, test_labels = parse_dataset(test_path, label_type)
    test_dataset = CTSweepDataset(test_imgs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for img, label in test_loader:
            output = model(img)
            y_pred.append(output.item())
            y_true.append(label.item())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    r, _ = pearsonr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='#011f5b')
    plt.xlabel('Ground Truth Angle (°)', fontsize=16)
    plt.ylabel('Predicted Angle (°)', fontsize=16)
    # plt.title('Correlation Plot for ' + label_type + ' Predictions', fontsize=14)
    
    if label_type.lower() == "pitch":
        plt.xlim(-17, 17)
        plt.ylim(-17, 17)
        plt.xticks(np.arange(-15, 16, 5), fontsize=14)
        plt.yticks(np.arange(-15, 16, 5), fontsize=14)
        lim = [-15, 15]
    else:  # assume roll
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.xticks(np.arange(-90, 91, 30), fontsize=14)
        plt.yticks(np.arange(-90, 91, 30), fontsize=14)
        lim = [-90, 90]

    # Plot the identity line with no label
    plt.plot(lim, lim, 'r', linestyle='--') #, label="Ideal Prediction (y = x)")

    # Add an invisible point just for the legend with r value
    #plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Pearson r: {r:.4f}")
    print(f"Test MAE: {mae:.2f}°")
    print(f"Mean Bias: {bias:.2f}°")


model_path = "/Users/emiz/Desktop/resnet18_smoothl1_PITCH3.pt"
test_path = "/Users/emiz/Desktop/ct_sweep_datasets/testing"
# bland_altman(model_path, test_path, "Roll")
# correlation_plot(model_path, test_path, "Pitch")