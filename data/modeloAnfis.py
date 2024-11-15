import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar el dataset desde la ruta especificada
data = pd.read_csv("data/combined_data_2016.csv")

# Preprocesamiento de datos - Usaremos un subconjunto de variables para simplificar el modelo
selected_features = ["Embalse_AMANI", "Embalse_GUAVIO", "Embalse_PENOL", "Embalse_PLAYAS"]
features = data[selected_features]
target = data["MPO"]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(features)
y = scaler_y.fit_transform(target.values.reshape(-1, 1))

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Clase de funciones de membresía
class MembershipFunctionLayer(nn.Module):
    def __init__(self, n_inputs, n_mf):
        super(MembershipFunctionLayer, self).__init__()
        self.n_mf = n_mf
        self.centers = nn.Parameter(torch.rand(n_inputs, n_mf))
        self.widths = nn.Parameter(torch.rand(n_inputs, n_mf))

    def forward(self, x):
        x_expanded = x.unsqueeze(2)
        centers_expanded = self.centers.unsqueeze(0)
        widths_expanded = self.widths.unsqueeze(0)
        mf_out = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * widths_expanded ** 2))
        
        return mf_out.reshape(x.size(0), -1)

# Capa de reglas
class RuleLayer(nn.Module):
    def __init__(self, n_rules):
        super(RuleLayer, self).__init__()
        self.weights = nn.Parameter(torch.rand(n_rules, 1))

    def forward(self, mf_out):
        # Mantener la segunda dimensión usando keepdim=True
        product = torch.prod(mf_out, dim=1, keepdim=True)
        
        # Ajuste de dimensiones para la multiplicación
        product = product.repeat(1, self.weights.size(0) // product.size(1))
        
        return torch.matmul(product, self.weights)

# Modelo ANFIS
class ANFIS(nn.Module):
    def __init__(self, n_inputs, n_mf):
        super(ANFIS, self).__init__()
        self.n_rules = n_mf ** n_inputs  # Total de reglas
        self.mf_layer = MembershipFunctionLayer(n_inputs, n_mf)
        self.rule_layer = RuleLayer(self.n_rules)

    def forward(self, x):
        mf_out = self.mf_layer(x)
        output = self.rule_layer(mf_out)
        return output

# Crear el modelo con un número reducido de funciones de membresía
n_inputs = X.shape[1]
n_mf = 2  # Reducir el número de funciones de membresía para limitar la complejidad
anfis_model = ANFIS(n_inputs, n_mf)

# Definir el optimizador y la función de pérdida
optimizer = torch.optim.Adam(anfis_model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Entrenamiento
n_epochs = 100
for epoch in range(n_epochs):
    optimizer.zero_grad()
    output = anfis_model(X)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Guardar el modelo
torch.save(anfis_model.state_dict(), "anfis_model.pth")
print("Modelo ANFIS guardado como 'anfis_model.pth'")
