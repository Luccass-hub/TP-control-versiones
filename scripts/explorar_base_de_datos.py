from pathlib import Path
import sys
import statsmodels.api as sm



RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.append(str(RUTA_PROYECTO))
from src.carga_datos import CargaDatos



COLUMNAS_NUMERICAS = [
    "Age",
    "MP",
    "Starts",
    "Min",
    "Gls",
    "Ast",
    "Sh",
    "SoT",
    "xG",
    "xAG",
    "Cmp",
    "Att",
    "PrgP",
    "Tkl",
    "Int",
    "90s",
    "G+A",
    "Sh/90",
    "Cmp%",
    "KP",
    "SCA",
    "PrgC",
    "Carries",
    "CrdY",
    "CrdR",
]

ruta_datos = RUTA_PROYECTO / "data" / "jugadores_2024_2025.csv"
cargador = CargaDatos(ruta_datos)


ruta_datos = RUTA_PROYECTO / "data" / "jugadores_2024_2025.csv"
cargador = CargaDatos(ruta_datos)

cargador.cargar_datos()
cargador.mostrar_dimensiones()

print()
cargador.mostrar_primeras_filas(n=5)

print()
cargador.mostrar_informacion()

print()
cargador.mostrar_valores_faltantes(n=10)

print()
cargador.mostrar_resumen_numerico(columnas=COLUMNAS_NUMERICAS)

#Debemos entrenar un modelo con Regresión Lineal Multiple, por lo tanto vamos a buscar las 
#Variables que resulten mas llamativas para hacer la regresion
#La variable respuesta es Gls

datos=cargador.cargar_datos()
print(datos.head(5))


