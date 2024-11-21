import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
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
    return anfis_model

# Generar predicciones anuales para otros años con ajuste mensual
def generar_predicciones_anuales(anfis_model, scaler_X, scaler_y, selected_features, year):
    if year in [2016, 2024]:
        print(f"Predicciones especiales ya generadas para febrero de {year}.")
        return

    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23)
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')

    # Factores de ajuste por mes (puedes personalizar estos valores)
    month_adjustments = {
        1: 0.95,  # Enero: 5% menos
        2: 1.00,  # Febrero: Sin ajuste
        3: 1.02,  # Marzo: 2% más
        4: 1.05,  # Abril: 5% más
        5: 1.03,  # Mayo: 3% más
        6: 0.98,  # Junio: 2% menos
        7: 1.04,  # Julio: 4% más
        8: 1.01,  # Agosto: 1% más
        9: 0.97,  # Septiembre: 3% menos
        10: 1.06,  # Octubre: 6% más
        11: 1.02,  # Noviembre: 2% más
        12: 1.00   # Diciembre: Sin ajuste
    }

    # Simular datos para cada característica
    embalse_means = {feature: 0.5 for feature in selected_features}
    simulated_data = {
        "FechaHora": date_range,
        "Mes": date_range.month,
        **{feature: [embalse_means[feature] * (1 + np.random.uniform(-0.05, 0.05)) for _ in date_range]
           for feature in selected_features}
    }
    simulated_df = pd.DataFrame(simulated_data)

    # Aplicar ajustes mensuales a las características simuladas
    for feature in selected_features:
        simulated_df[feature] *= simulated_df["Mes"].map(month_adjustments)

    # Escalar los datos simulados
    X_future = scaler_X.transform(simulated_df[selected_features])
    X_future_tensor = torch.tensor(X_future, dtype=torch.float32)

    # Generar predicciones con el modelo
    with torch.no_grad():
        y_future_scaled = anfis_model(X_future_tensor).detach().cpu().numpy()
        y_future = scaler_y.inverse_transform(y_future_scaled)

    # Aplicar ajuste mensual a las predicciones
    simulated_df["Prediccion_MPO"] = y_future.flatten() * simulated_df["Mes"].map(month_adjustments)

    # Guardar las predicciones en un archivo CSV
    simulated_df[["FechaHora", "Prediccion_MPO"]].to_csv(f"data/predicciones_mpo_{year}.csv", index=False)
    print(f"Predicciones guardadas para el año {year} en 'data/predicciones_mpo_{year}.csv'.")

# Entrenamiento y generación de predicciones
X, y, scaler_X, scaler_y, selected_features = cargar_datos()
anfis_model = entrenar_modelo(X, y, n_inputs=X.shape[1], n_mf=2, n_epochs=200, batch_size=32, lr=0.005)
generar_predicciones_anuales(anfis_model, scaler_X, scaler_y, selected_features, 2025)
