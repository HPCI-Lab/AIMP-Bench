import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('benchmark_params/gpu_dataset_ts.csv', sep=";")
for col in ["GPU Avg Power Consumption (W)"]: 
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[~((df[col] < lower_bound) | (df[col] > upper_bound))]

df_encoded = pd.get_dummies(df, columns=['GPU', "GPU Memory Request (Gb)","Walltime (min)"])

start_w = df["GPU Power Consumption Start (W)"].map(lambda x: eval(x))
start_w = pd.DataFrame.from_dict(dict(zip(start_w.index, start_w.values))).T

X = df_encoded.drop([
    'GPU Avg Power Consumption (W)', 
    "GPU Power Consumption Start (W)", 
    "GPU Power Consumption All (W)", 
    "Unnamed: 0", 
    ], axis=1)
# start_w = start_w.mean(axis=1).rename('new')
X = X.join(start_w)
X = X.astype(float)
X.columns = X.columns.astype(str) 
y = df_encoded['GPU Avg Power Consumption (W)']

for col in X.columns: # Norm cols
    X[col] = (X[col] - min(X[col])) / (max(X[col]) - min(X[col]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# model = linear_model.BayesianRidge()
model = MLPRegressor(warm_start=True, max_iter=200, tol=0.1, learning_rate="adaptive")
model = model.fit(X_train, y_train)

predictions = model.predict(X_train)
mae = mean_absolute_error(y_train, predictions)
r2 = r2_score(y_train, predictions)

print(f"--- Train Set Evaluation ---")
print(f"Mean Absolute Error: {mae:.2f} Watts")
print(f"R^2 Score: {r2:.2f}")

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"--- Test Set Evaluation ---")
print(f"Mean Absolute Error: {mae:.2f} Watts")
print(f"R^2 Score: {r2:.2f}")
