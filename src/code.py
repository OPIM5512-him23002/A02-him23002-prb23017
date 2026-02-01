from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
X = housing.frame.drop(columns=["MedHouseVal"])
y = housing.frame["MedHouseVal"]
                  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(random_state=42,
                   hidden_layer_sizes=(10,5),
                   alpha=1e-3,
                   max_iter=200,
                   batch_size=1000,
                   activation="relu",
                   validation_fraction=0.2,
                   early_stopping=True) 
mlp.fit(X_train_scaled, y_train)