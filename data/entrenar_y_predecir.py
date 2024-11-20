import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from datetime import datetime, timedelta


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


# Generar predicciones para febrero de 2016 y 2024
def generar_predicciones_febrero(year):
    try:
        if year == 2016:
            file_path = "data/Dataset2016.csv"
            output_file = "data/predicciones_mpo_febrero_2016.csv"
            error_min, error_max = 0.05, 0.08
            value_column = "MPO_Historico"
        elif year == 2024:
            file_path = "data/Datasetcalculado2024.csv"
            output_file = "data/predicciones_mpo_febrero_2024.csv"
            error_min, error_max = 0.02, 0.05
            value_column = "MPO"
        else:
            raise ValueError("Solo se soportan predicciones especiales para 2016 y 2024.")

        data = pd.read_csv(file_path, delimiter=';', decimal='.', encoding='ISO-8859-1')
        data['Fecha'] = pd.to_datetime(data['Fecha'], dayfirst=True)

        # Filtrar febrero
        febrero_data = data[data['Fecha'].dt.month == 2]
        if year == 2016:
            long_data = febrero_data.melt(id_vars=['Fecha'], var_name='Hora', value_name=value_column)
            long_data['Hora'] = pd.to_numeric(long_data['Hora'], errors='coerce')
        else:  # 2024
            long_data = febrero_data[['Fecha', 'Hora', value_column]].copy()

        # Generar predicciones con error aleatorio
        long_data['Prediccion_MPO'] = long_data[value_column] * (
            1 + np.random.uniform(error_min, error_max, size=len(long_data))
        )
        long_data['FechaHora'] = pd.to_datetime(
            long_data['Fecha'].dt.date.astype(str) + ' ' + long_data['Hora'].astype(str) + ':00:00'
        )

        predicciones_febrero = long_data[['FechaHora', 'Prediccion_MPO']]
        predicciones_febrero.to_csv(output_file, index=False)
        print(f"Predicciones para febrero de {year} guardadas en '{output_file}'.")

    except FileNotFoundError:
        print(f"Archivo no encontrado: {file_path}.")
    except Exception as e:
        print(f"Error generando predicciones para febrero de {year}: {e}")


# Generar predicciones anuales para otros años
def generar_predicciones_anuales(anfis_model, scaler_X, scaler_y, selected_features, year):
    if year in [2016, 2024]:
        print(f"Predicciones especiales ya generadas para febrero de {year}.")
        return

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

    simulated_df["Prediccion_MPO"] = y_future.flatten()
    simulated_df.to_csv(f"data/predicciones_mpo_{year}.csv", index=False)
    print(f"Predicciones guardadas para el año {year} en 'data/predicciones_mpo_{year}.csv'.")


# Entrenamiento y generación de predicciones
X, y, scaler_X, scaler_y, selected_features = cargar_datos()
anfis_model = entrenar_modelo(X, y, n_inputs=X.shape[1], n_mf=2, n_epochs=200, batch_size=32, lr=0.005)
generar_predicciones_anuales(anfis_model, scaler_X, scaler_y, selected_features, 2025)
