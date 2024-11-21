import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from datetime import datetime

# Función para cargar datos
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

# Modelo ANFIS
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
    return anfis_model, val_loader

# Evaluar modelo
def evaluar_modelo(anfis_model, val_loader, scaler_y):
    anfis_model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = anfis_model(batch_X).detach().cpu().numpy()
            targets = batch_y.detach().cpu().numpy()
            predictions = scaler_y.inverse_transform(predictions)
            targets = scaler_y.inverse_transform(targets)
            all_predictions.extend(predictions.flatten())
            all_targets.extend(targets.flatten())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    mae = mean_absolute_error(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    
    print("=== Métricas de Evaluación ===")
    print(f"Error Absoluto Medio (MAE): {mae:.4f}")
    print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
    print(f"Coeficiente de Determinación (R²): {r2:.4f}")
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
}

# Generar predicciones anuales para otros años con ajuste mensual
def generar_predicciones_anuales(anfis_model, scaler_X, scaler_y, selected_features, year):
    # (Código de generación de predicciones permanece igual)
    pass

# Ejecución principal
X, y, scaler_X, scaler_y, selected_features = cargar_datos()
anfis_model, val_loader = entrenar_modelo(X, y, n_inputs=X.shape[1], n_mf=2, n_epochs=200, batch_size=32, lr=0.005)
metricas = evaluar_modelo(anfis_model, val_loader, scaler_y)
