import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime

# Función para cargar y preparar datos
def cargar_datos(seq_length=24):
    data = pd.read_csv("data/combined_data_2016.csv")
    selected_features = [col for col in data.columns if col.startswith("Embalse_")]
    features = data[selected_features]
    target = data["MPO"]

    # Escalado de características
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(features)
    y_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

    # Crear secuencias para LSTM
    X_seq, y_seq = crear_secuencias(X_scaled, y_scaled, seq_length)

    X_seq = torch.tensor(X_seq, dtype=torch.float32)
    y_seq = torch.tensor(y_seq, dtype=torch.float32)

    return X_seq, y_seq, scaler_X, scaler_y, selected_features

# Función para crear secuencias de tiempo
def crear_secuencias(X, y, seq_length):
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

# Definir el modelo LSTM
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Tomar la última salida del LSTM
        out = self.fc(out)
        return out

# Función para entrenar el modelo
def entrenar_modelo(X, y, n_epochs=50, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_size = X.shape[2]
    model = LSTMModel(input_size).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 5 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_X)
                    val_loss = criterion(outputs, batch_y)
                    val_losses.append(val_loss.item())
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}, Val Loss: {np.mean(val_losses):.6f}")

    torch.save(model.state_dict(), "data/lstm_model.pth")
    print("Modelo LSTM guardado como 'lstm_model.pth'")
    return model, device

# Generar predicciones anuales para 2025
def generar_predicciones_anuales(model, device, scaler_X, scaler_y, selected_features, year, seq_length=24):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')

    # Simular datos para las características exógenas
    embalse_means = {feature: 0.5 for feature in selected_features}
    simulated_data = {
        **{feature: [embalse_means[feature] * (1 + np.random.uniform(-0.05, 0.05)) for _ in range(len(date_range))]
           for feature in selected_features}
    }
    simulated_df = pd.DataFrame(simulated_data, index=date_range)

    # Escalar las características
    X_future_scaled = scaler_X.transform(simulated_df[selected_features])

    # Crear secuencias para predicción
    X_future_seq = []
    for i in range(len(X_future_scaled) - seq_length):
        X_future_seq.append(X_future_scaled[i:i+seq_length])
    X_future_seq = np.array(X_future_seq)
    X_future_seq = torch.tensor(X_future_seq, dtype=torch.float32).to(device)

    # Realizar predicciones
    model.eval()
    with torch.no_grad():
        y_future_scaled = model(X_future_seq).cpu().numpy()

    # Invertir el escalado de las predicciones
    y_future = scaler_y.inverse_transform(y_future_scaled)

    # Crear DataFrame de predicciones
    pred_dates = date_range[seq_length:]
    pred_df = pd.DataFrame({
        "FechaHora": pred_dates,
        "Prediccion_MPO": y_future.flatten()
    })

    # Guardar las predicciones
    pred_df.to_csv(f"data/predicciones_mpo_{year}.csv", index=False)
    print(f"Predicciones guardadas para el año {year} en 'data/predicciones_mpo_{year}.csv'.")

# Ejecución del flujo completo
seq_length = 24  # Por ejemplo, utilizamos secuencias de 24 horas
X_seq, y_seq, scaler_X, scaler_y, selected_features = cargar_datos(seq_length=seq_length)
model, device = entrenar_modelo(X_seq, y_seq, n_epochs=50, batch_size=64, lr=0.001)
generar_predicciones_anuales(model, device, scaler_X, scaler_y, selected_features, 2025, seq_length=seq_length)
