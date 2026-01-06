import pandas as pd 
import numpy as np
from pathlib import Path 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error, r2_score 
import joblib

def load_data():
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / "CSV_trabajos_final.csv"
    return pd.read_csv(csv_path)

def prepare_features(df):
    #se selecciona solo las columnas deseadas
    X = df[['experiencia']]
    y = df['salario']

    #codificar la experiencia
    encoder = OrdinalEncoder(categories=[['junior', 'mid', 'senior']])
    X_encoded = encoder.fit_transform(X)

    return X_encoded, y, encoder

def train_GridSearch(X_train, y_train):
    #modelos base
    lr = LinearRegression()
    dt = DecisionTreeRegressor(random_state=42)

    #hiperparametros para DecisionTree
    param_grid_dt = {
        "max_depth": [2, 3, 4, 5, 6, None],
        "min_samples_split": [2, 5, 10]
    }
    
    #GridSearch para DecisionTree
    grid_dt = GridSearchCV(
        estimator = dt,
        param_grid = param_grid_dt,
        scoring = "neg_mean_squared_error",
        cv = 5,
        n_jobs = -1
    )

    #Entrenar modelos
    lr.fit(X_train, y_train)
    grid_dt.fit(X_train, y_train)

    #mejor modelo del arbol
    best_dt = grid_dt.best_estimator_


    return lr, best_dt, grid_dt.best_estimator_

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

if __name__ == "__main__":
    #cargar datos
    df = load_data()

    #preparar features
    X, y, encoder = prepare_features(df)

    #train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42
    )

    #entrenar modelos
    lr_model, dt_model, dt_best_params = train_GridSearch(X_train, y_train)

    # evaluar
    lr_mse, lr_r2 = evaluate_model(lr_model, X_test, y_test)
    dt_mse, dt_r2 = evaluate_model(dt_model, X_test, y_test)

    # comparar y elegir cual es mejor
    if dt_mse < lr_mse:
        best_model = dt_model
        best_name = "DecisionTreeRegressor"
    else:
        best_model = lr_model
        best_name = "LinearRegression"

    #guardar modelo
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    joblib.dump(best_model, models_dir / "modelo_salarios.joblib")

    #guardar metricas
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    metrics_df = pd.DataFrame({
        "Modelo": ["LinearRegression", "DecisionTreeRegressor"],
        "MSE": [lr_mse, dt_mse],
        "R2": [lr_r2, dt_r2]
    })

    metrics_df.to_csv(results_dir / "metricas_ml.csv", index=False)

    #resultados:
    print("Evaluacion de los modelos: ")
    print(metrics_df)

    print("\nModelo seleccionado:", best_name)
    print("\nMejores hiperparámetros del árbol:", dt_best_params)