import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from datetime import datetime, timedelta

import os
import subprocess

# Instalar PyTorch si no está disponible
try:
    import torch
except ImportError:
    subprocess.check_call(
        ["pip", "install", "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"]
    )
    import torch


def cargar_datos():
    data = pd.read_csv("data/combined_data_2016.csv")
    selected_features = [col for col in data.columns if col.startswith("Embalse_")]
    features = data[selected_features]
    target = data["MPO"]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(features)
    y = scaler_y.fit_transform(target.values.reshape(-1, 1))
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y, scaler_X, scaler_y, selected_features

def entrenar_modelo(X, y, n_inputs, n_mf=2, n_epochs=200, batch_size=32, lr=0.005):
    class MembershipFunctionLayer(torch.nn.Module):
        def __init__(self, n_inputs, n_mf):
            super().__init__()
            self.centers = torch.nn.Parameter(torch.rand(n_inputs, n_mf))
            self.widths = torch.nn.Parameter(torch.rand(n_inputs, n_mf))

        def forward(self, x):
            x_expanded = x.unsqueeze(2)
            centers_expanded = self.centers.unsqueeze(0)
            widths_expanded = self.widths.unsqueeze(0)
            mf_out = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * widths_expanded ** 2))
            return mf_out.reshape(x.size(0), -1)

    class RuleLayer(torch.nn.Module):
        def __init__(self, mf_output_size):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.rand(mf_output_size, 1))

        def forward(self, mf_out):
            return torch.matmul(mf_out, self.weights)

    class ANFIS(torch.nn.Module):
        def __init__(self, n_inputs, n_mf):
            super().__init__()
            self.mf_layer = MembershipFunctionLayer(n_inputs, n_mf)
            mf_output_size = n_inputs * n_mf
            self.rule_layer = RuleLayer(mf_output_size)

        def forward(self, x):
            mf_out = self.mf_layer(x)
            output = self.rule_layer(mf_out)
            return output

    anfis_model = ANFIS(n_inputs, n_mf)
    optimizer = torch.optim.Adam(anfis_model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    train_size = int(0.8 * len(X))
    val_size = len(X) - train_size
    train_dataset, val_dataset = random_split(TensorDataset(X, y), [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(n_epochs):
        anfis_model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = anfis_model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            anfis_model.eval()
            val_loss = sum(loss_fn(anfis_model(batch_X), batch_y) for batch_X, batch_y in val_loader) / len(val_loader)
            print(f"Epoch {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    torch.save(anfis_model.state_dict(), "data/anfis_model.pth")
    print("Modelo ANFIS guardado como 'anfis_model.pth'")
    return anfis_model

# Postentrenamiento y desnormalización
def desnormalizar_predicciones(y_pred, scaler_y):
    return scaler_y.inverse_transform(y_pred)

def evaluar_modelo_postentrenamiento(modelo, X_test, y_test, scaler_y):
    modelo.eval()
    with torch.no_grad():
        y_pred = modelo(X_test).numpy()
        y_pred = desnormalizar_predicciones(y_pred, scaler_y)
        y_test = desnormalizar_predicciones(y_test.numpy(), scaler_y)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return y_pred, mse, mae, r2

def clip_predicciones(y_pred, min_val=0, max_val=1000):
    return np.clip(y_pred, min_val, max_val)

def suavizar_predicciones(y_pred, ventana=5):
    return pd.Series(y_pred.flatten()).rolling(window=ventana, min_periods=1).mean().values

# Predicciones anuales
def generar_predicciones_anuales(anfis_model, scaler_X, scaler_y, selected_features, year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23)
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')

    embalse_means = {feature: 0.5 for feature in selected_features}
    simulated_data = {
        "FechaHora": date_range,
        **{feature: [embalse_means[feature] * (1 + np.random.uniform(-0.05, 0.05)) for _ in date_range]
           for feature in selected_features}
    }
    simulated_df = pd.DataFrame(simulated_data)
    X_future = scaler_X.transform(simulated_df[selected_features])
    X_future_tensor = torch.tensor(X_future, dtype=torch.float32)

    with torch.no_grad():
        y_future_scaled = anfis_model(X_future_tensor).numpy()
        y_future = scaler_y.inverse_transform(y_future_scaled)

    simulated_df["Prediccion_MPO"] = suavizar_predicciones(clip_predicciones(y_future))
    simulated_df.to_csv(f"data/predicciones_mpo_{year}.csv", index=False)
    print(f"Predicciones guardadas para el año {year} en 'data/predicciones_mpo_{year}.csv'")

# Entrenamiento y evaluación
X, y, scaler_X, scaler_y, selected_features = cargar_datos()
anfis_model = entrenar_modelo(X, y, n_inputs=X.shape[1], n_mf=2, n_epochs=200, batch_size=32, lr=0.005)
y_pred, mse, mae, r2 = evaluar_modelo_postentrenamiento(anfis_model, X, y, scaler_y)

# Generar predicciones para los años deseados
for year in [2016, 2024, 2025]:
    generar_predicciones_anuales(anfis_model, scaler_X, scaler_y, selected_features, year)
