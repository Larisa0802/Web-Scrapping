import requests
from bs4 import BeautifulSoup
import csv
import re

url = "https://www.tecnoempleo.com/busqueda-empleo.php?te=desarrollador+web"
headers = {"User-Agent": "Mozilla/5.0"}

html = requests.get(url, headers=headers).text
soup = BeautifulSoup(html, "html.parser")

lineas = soup.get_text(separator="\n", strip=True).split("\n")

ofertas = []

def es_titulo_valido(texto):
    if "€" in texto:
        return False
    if re.search(r"\d{2}/\d{2}/\d{4}", texto):
        return False
    if "(" in texto:
        return False
    palabras_clave = [
        "Desarrollador", "Programador", "Web", "Java",
        "Frontend", "Backend", "Fullstack", "Full Stack"
    ]
    return any(p.lower() in texto.lower() for p in palabras_clave)


for i, linea in enumerate(lineas):
    if "€" in linea:
        try:
            salario = linea
            posible_titulo = lineas[i-5]

            if not es_titulo_valido(posible_titulo):
                continue

            titulo = posible_titulo
            empresa = lineas[i-4]
            ubicacion = lineas[i-3]
            modalidad_fecha = lineas[i-2]

            ofertas.append([
                titulo,
                empresa,
                ubicacion,
                modalidad_fecha,
                salario
            ])
        except IndexError:
            continue


with open("ofertas_tecnoempleo.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["titulo", "empresa", "ubicacion", "modalidad_fecha", "salario"])
    writer.writerows(ofertas)

print("Scraping terminado.")
print("Ofertas guardadas:", len(ofertas))
