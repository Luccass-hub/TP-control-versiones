import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.append(str(RUTA_PROYECTO))

from src.carga_datos import CargaDatos
ruta_datos = RUTA_PROYECTO / "data" / "jugadores_2024_2025.csv"
cargador = CargaDatos(ruta_datos)
df = cargador.cargar_datos()

# Regresion Lineal Simple.
'''
En esta primera parte, debemos predecir la cantidad de goles mediante un modelo de regresion
lineal simple. Para hacerlo utlizando una unica variable, tomaremos las variables Sh (tiros totales), SoT (tiros al arco) 
y Min (minutos jugados) para hacer, por separado, cada uno de los modelos.
Decidimos utilizar estas variables porque supusimos que, a mayor cantidad de tiros (totales y al arco), mas probable es 
que hagan goles. El mismo razonamiento utilizamos para seleccionar minutos jugados (a mayor tiempo jugado, mayor posibilidad de meter un gol).
'''

# df = df[df['Gls'] > 0]
y = df['Gls']

'''
# Variable predictora: tiros totales
x = np.array(df['Sh'])
X = sm.add_constant(x)
modelo1 = sm.OLS(y, X)
result1 = modelo1.fit()
print(result1.summary())

plt.figure()
sns.scatterplot(data=df, x='Sh', y=y)
plt.title('Tiros totales')
plt.show()

# Variable predictora: tiros al arco
x = np.array(df['SoT'])
X = sm.add_constant(x)
modelo2 = sm.OLS(y, X)
result2 = modelo2.fit()
print(result2.summary())

plt.figure()
sns.scatterplot(data=df, x='SoT', y=y)
plt.title('Tiros al arco')
plt.show()

# Variable predictora: minutos jugados
x = np.array(df['Min'])
X = sm.add_constant(x)
modelo3 = sm.OLS(y, X)
result3 = modelo3.fit()
print(result3.summary())

plt.figure()
sns.scatterplot(data=df, x='Min', y=y)
plt.title('Minutos jugados')
plt.show()


Comparando los R-ajustados de cada modelo, vemos que tiros al arco (0.726) y tiros totales (0.814) se ajustan, siendo este último la mejor
de las tres variables. Mientras que minutos jugados no; con un R-ajustado de 0.162. 
'''

print((df['Gls'] !=0).sum())
