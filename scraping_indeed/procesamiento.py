import pandas as pd
import numpy as np
import re
from pathlib import Path


def parse_salary(salary_raw):
    """
    Convierte salarios tipo:
    '36.000€ - 39.000€ b/a' -> 37500
    """
    if pd.isna(salary_raw):
        return np.nan

    # Quitar puntos y extraer números
    numbers = re.findall(r'\d+', salary_raw.replace('.', ''))

    if len(numbers) == 0:
        return np.nan
    elif len(numbers) == 1:
        salary = float(numbers[0])
    else:
        salary = np.mean([float(numbers[0]), float(numbers[1])])

    return salary


def experience_from_salary(salary_series):
    """
    Calcula la experiencia en base al salario:
    < 30000  -> 0 (Junior)
    30000-40000 -> 1 (Mid)
    > 40000 -> 2 (Senior)
    """
    conditions = [
        salary_series < 30000,
        (salary_series >= 30000) & (salary_series <= 40000),
        salary_series > 40000
    ]
    choices = [0, 1, 2]

    return np.select(conditions, choices)


def clean_dataset(df):
    df = df.copy()

    # Procesar salario
    df['salary'] = df['salario'].apply(parse_salary)

    # Imputar salarios faltantes con la mediana
    salary_median = np.nanmedian(df['salary'])
    df['salary'] = df['salary'].fillna(salary_median)

    # Calcular experiencia a partir del salario
    df['experience'] = experience_from_salary(df['salary'])

    # Dataset final
    df_final = df[['titulo', 'salary', 'experience']]
    df_final.columns = ['titulo', 'salario', 'experiencia']

    return df_final


if __name__ == "__main__":
    # Cargar CSV bruto
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / "ofertas_tecnoempleo.csv"
    df_raw = pd.read_csv(csv_path)

    # Limpieza y transformación
    df_clean = clean_dataset(df_raw)


    # Convertir `salario` a entero y mapear `experiencia` a etiquetas
    df_clean['salario'] = df_clean['salario'].astype(int)
    mapping = {0: 'junior', 1: 'mid', 2: 'senior'}
    df_clean['experiencia'] = df_clean['experiencia'].map(mapping)

    # Asegurar el orden de columnas y guardar CSV final
    df_clean = df_clean[['titulo', 'salario', 'experiencia']]
    output_path = script_dir / "CSV_trabajos_final.csv"
    df_clean.to_csv(output_path, index=False)

    print("CSV_trabajos_final.csv generado correctamente.")
