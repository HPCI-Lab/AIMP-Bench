import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('benchmark_params/gpu_dataset_ts.csv', sep=";")
print(len(df))
for col in ["GPU Avg Power Consumption (W)"]: 
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[~((df[col] < lower_bound) | (df[col] > upper_bound))]
print(len(df))

df_encoded = pd.get_dummies(df, columns=['GPU', "GPU Memory Request (Gb)","Walltime (min)"])

start_w = df["GPU Power Consumption Start (W)"].map(lambda x: eval(x))
start_w = pd.DataFrame.from_dict(dict(zip(start_w.index, start_w.values))).T

X = df_encoded.drop([
    'GPU Avg Power Consumption (W)', 
    "GPU Power Consumption Start (W)", 
    "GPU Power Consumption All (W)", 
    "Unnamed: 0", 
    ], axis=1)
X = X.join(start_w).astype(float)
X.columns = X.columns.astype(str) 
y = df_encoded['GPU Avg Power Consumption (W)']

for col in X.columns: # Norm cols
    X[col] = (X[col] - min(X[col])) / (max(X[col]) - min(X[col]))

X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2)

# Create DataLoaders for batching
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# --- 2. Define a Simple Forecaster Model ---
class PowerForecaster(nn.Module):
    def __init__(self, input_size):
        super(PowerForecaster, self).__init__()
        # A simple 3-layer architecture
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Single output for Avg Power
        )
        
    def forward(self, x):
        return self.layers(x)

def get_avg_loss(test_loader): 
    test_loss = 0.0
    test_total = 0
    with torch.no_grad():
        for X_test, y_test in test_loader: 
            test_preds = model(X_test)
            test_loss += criterion(test_preds, y_test).item()
            test_total += X_test.shape[0]
    return test_loss / test_total

# Initialize model, loss, and optimizer
model = PowerForecaster(X.shape[1])
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 3. Training Loop ---
epochs = 1000
model.train()

losses = []
patience = 15
min_val_loss = 100000
for epoch in tqdm(range(epochs)):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        losses.append(loss.item())

    val_loss = get_avg_loss(test_loader)
    if min_val_loss > val_loss: 
        min_val_loss = val_loss
        patience = 5 
    else: 
        patience -= 1
    if patience == 0: 
        break

    
plt.plot(losses)
plt.show()

model.eval()
train_loss = get_avg_loss(train_loader)
test_loss = get_avg_loss(test_loader)
print(f"{round(train_loss, 2)}", f"{round(test_loss, 2)}")