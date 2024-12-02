# %%
'''
Finger flexion prediction using CNN-LSTM
@autor: Deng Chijun
@create date: 2024.11.24
@update date: 2024.11.27
'''
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import StepLR
import pickle

from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# %%
'''
CNN-LSTM Regressor
'''


class CNN_LSTM_Regressor(nn.Module):
    def __init__(self, input_dim, cnn_channels, lstm_hidden_dim, lstm_num_layers, output_dim, dropout=0.2):
        super(CNN_LSTM_Regressor, self).__init__()
        # CNN部分
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM部分
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden_dim, lstm_num_layers, batch_first=True, dropout=dropout)

        # 全连接层
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, freq, channels)
        # Permute to (batch_size, channels, freq) for CNN
        # x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # (batch_size, cnn_channels, freq/2)

        # Permute back to (batch_size, seq_length, input_dim) for LSTM
        x = x.permute(0, 2, 1)

        # LSTM
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)

        # Aggregate over the sequence dimension
        out = out.mean(dim=1)  # (batch_size, hidden_size)
        out = self.fc(out)  # (batch_size, output_dim)
        return out


# %%
'''
Train and test for each subject and finger
'''
# Preprocessed data directory (feature)
root_dir = 'E:/Code/Class_Project/BCI_Prj_New/dataset'
process_dir = f"{root_dir}/BCI_Competion4_dataset4_data_fingerflexions/c_preprocessing"

# Results directory
pred_dir = f"{root_dir}/BCI-Finger-Flex/prediction/cnn_lstm"
model_dir = f"{root_dir}/BCI-Finger-Flex/model/cnn_lstm"

# Gaussian filter sigma
sigma = 6

# Subjects and fingers
subs = [1]
fingers = ['thumb', 'index', 'middle', 'ring', 'little']

for idx, subid in enumerate(subs):
    '''
    Data loading
    '''
    data_process_dir = f'{process_dir}/sub{subid}'

    # train
    X_train = np.load(f'{data_process_dir}/train/ecog_data.npy')
    y_train_all = np.load(f'{data_process_dir}/train/fingerflex_data.npy')

    # test
    X_test = np.load(f'{data_process_dir}/val/ecog_data.npy')
    y_test_all = np.load(f'{data_process_dir}/val/fingerflex_data.npy')

    # Transpose to (times, channels, wavelets)
    X_train = X_train.transpose(2, 0, 1)
    X_test = X_test.transpose(2, 0, 1)
    y_train_all = y_train_all.transpose(1, 0)
    y_test_all = y_test_all.transpose(1, 0)

    # Feature dim
    num_wave = X_train.shape[2]
    num_node = X_train.shape[1]

    '''
    Model Setup
    '''
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model = CNN_LSTM_Regressor(input_dim=num_node, cnn_channels=64, lstm_hidden_dim=128,
                               lstm_num_layers=2, output_dim=5, dropout=0.15).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    '''
    Save dir
    '''
    # Model save dir
    model_finger_dir = f"{model_dir}/sub{subid}"
    if not os.path.exists(model_finger_dir):
        os.makedirs(model_finger_dir)
        logging.info(f"Directory '{model_finger_dir}' created.")

    # Prediction save dir
    pred_finger_dir = f"{pred_dir}/sub{subid}"
    if not os.path.exists(pred_finger_dir):
        os.makedirs(pred_finger_dir)
        logging.info(f"Directory '{pred_finger_dir}' created.")

    # for finger_id, finger_name in enumerate(fingers):

    finger_name = "ALL"
    # logging train and validation process
    logging.info(f"Subject {subid} {finger_name} flexion prediction using CNN-LSTM regressor")
    '''
    Torch dataloader
    '''
    # y_train = y_train_all[:, finger_id].reshape([-1, 1])
    # y_test = y_test_all[:, finger_id].reshape([-1, 1])

    y_train = y_train_all
    y_test = y_test_all

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train epoch
    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), f"{model_finger_dir}/{finger_name}.pth")

    '''
    Evaluation and Inference
    '''
    y_pred = []
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            test_loss += loss.item()

            y_pred.append(predictions.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)  # Combine batch predictions

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

    '''
    Calculating Pearson correlation, MSE metrics
    '''
    for finger_id, finger_name in enumerate(fingers):

        y_pred_filter = gaussian_filter1d(y_pred[:,finger_id].ravel(), sigma=sigma)
        rho, pval = pearsonr(y_test[:,finger_id].ravel(), y_pred_filter)
        mse = mean_squared_error(y_test[:,finger_id].ravel(), y_pred_filter)

        # Save prediction and metrics
        pred_res = {
            'y_test': y_test[:,finger_id].ravel(),
            'y_pred': y_pred[:,finger_id].ravel(),
            'y_pred_filter': y_pred_filter.ravel(),
            'rho': rho,
            'pval': pval,
            'mse': mse
        }
        with open(f"{pred_finger_dir}/{finger_name}.pkl", 'wb') as pickle_file:
            pickle.dump(pred_res, pickle_file)

        '''
        Visualization
        '''
        sns.set_theme(style='darkgrid', font_scale=1.2)
        colors = sns.color_palette("Set2", 2)

        plt.figure(figsize=(12, 6))
        # Plot original y
        plt.plot(y_test[:,finger_id].ravel(), label="Ground truth", alpha=0.7, color=colors[0])
        # Plot predicted y
        plt.plot(y_pred_filter.ravel(), label="Prediction", alpha=0.7, color=colors[1])

        # Adding labels and legend
        plt.title(
            f"Prediction of {finger_name} flexion of subject {subid} using CNN-LSTM regressor (rho = {rho:.3f}, MSE = {mse:.3f})",
            fontsize=16)
        plt.xlabel("Time", fontsize=16)
        plt.ylabel("Signal", fontsize=16)
        plt.legend(fontsize=14, loc='upper right')
        plt.grid(True)
        plt.tight_layout()

        # Save figure
        plt.savefig(f"{pred_finger_dir}/{finger_name}.png", bbox_inches='tight', dpi=300)
        plt.show()