import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


#Cargar CSV de ofertas


csv_path = Path(__file__).resolve().parent / "scraping_indeed" / "CSV_trabajos_final.csv"
df = pd.read_csv(csv_path)


#BAR PLOT: salario medio por experiencia


salary_by_exp = df.groupby("experiencia")["salario"].mean().sort_values() #Agrupa el salario por experiencia (junior,mid,senior)

plt.figure(figsize=(7,5))
plt.bar(salary_by_exp.index, salary_by_exp.values, color=["#6baed6", "#3182bd", "#08519c"]) #Genera las lineas del grafico
plt.title("Salario medio por nivel de experiencia")
plt.xlabel("Nivel de experiencia")
plt.ylabel("Salario medio (€)")
plt.grid(axis="y", linestyle="--", alpha=0.4) #Cuadricula vertical
plt.tight_layout() #Recoloca elementos
plt.show()


#Preparar datos igual que en model.py

X = df[['experiencia']]
y = df['salario']

encoder = OrdinalEncoder(categories=[['junior', 'mid', 'senior']]) #Convierte la col. exp en números, de 0-2
X_encoded = encoder.fit_transform(X)

#Divide datos en 2 partes, una entrena al modelo (x/y_train) 80% y otra la evalua 20%
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)


#Cargar modelo entrenado

model_path = Path(__file__).resolve().parent / "models" / "modelo_salarios.joblib"
modelo = load(model_path)


#Generar predicciones reales

#Le pasas al modelo los datos del test x y devuelve las predicciones en y (el punto verde)
y_pred = modelo.predict(X_test)


#SCATTER PLOT: reales vs predichos

plt.figure(figsize=(6,6)) #Gráfico de 6x6
plt.scatter(y_test, y_pred, alpha=0.6, color="#2ca25f", edgecolor="k") #Dibuja un scatter con y_test y y_pred

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2) #Salario real = salario predicho
plt.title("Salarios reales vs predichos")
plt.xlabel("Salario real (€)")
plt.ylabel("Salario predicho (€)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
