from pathlib import Path
import sys
import statsmodels.api as sm
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
import pandas as pd
import seaborn as sns



RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.append(str(RUTA_PROYECTO))
from src.carga_datos import CargaDatos

ruta_datos = RUTA_PROYECTO / "data" / "jugadores_2024_2025.csv"
cargador = CargaDatos(ruta_datos)

df=cargador.cargar_datos()

#Debemos entrenar un modelo con Regresión Lineal Multiple, por lo tanto vamos a buscar las 
#Variables que resulten mas llamativas para hacer la regresion
#La variable respuesta es Gls
#print(df.columns)

#vamos a dar una regresión lineal multiple con las siguientes predictoras:
#tiros totales, tiros al arco, minutos jugados(ver si partidos jugados y tiros por cada 90 minutos  es significativa)
#Posicion, Equipo, minutos jugados,competencia o liga (veamos de quitar al arquero de posicion)
#Tarjetas rojas,acciones que generan tiros 


y=df["Gls"]
X_1=sm.add_constant(df[["Sh","SoT","MP","Sh/90","90s"]])
modelo_1=sm.OLS(y,X_1).fit()

print("Modelo N°1\n")
print(modelo_1.summary())
print("\n\n\n")


#Como vimos, este modelo tiene variables predictoras con p valores menores a alpha=0.05, fuera de la variable predictora tiros por cada 90 minutos.
#Vamos a crear un nuevo modelo sin esta predictora, y vamos a comparar si este modelo es mejor en base a anova.

X_2=sm.add_constant(df[["Sh","SoT","90s","MP"]])
modelo_2=sm.OLS(y,X_2).fit()

print("Modelo N°2\n")
print(modelo_2.summary())
print("\n\n\n")


#Vemos que el p-valor del parametro betha_0 siemore nos da muy grande,
#intuimos que esto se debe a que tenemos muchos valores en la variable resupuesta 
# que son cero.
# Lo que vamos a hacer ahora va a ser dar el modelo que nos resulto con mayor R_squared 
# solo coinsiderando los valores del vector Gls que no son nulos. 

df_validos=df[df["Gls"]!=0]

y_val=df_validos["Gls"]
X_3=sm.add_constant(df_validos[["Sh","SoT","90s","MP"]] )

modelo_3=sm.OLS(y_val,X_3).fit()

print("Modelo N°3 sin valores de goles nulos\n")
print(modelo_3.summary())
print("\n\n\n")


#Modelo sin ceros en Gls solo con predictoras con p-valor<0.05

X_4=sm.add_constant(df_validos[["SoT","90s","MP"]])
modelo_4=sm.OLS(y_val,X_4).fit()

print("Modelo N°4, sin nulos con  ")
print(modelo_4.summary())
print("\n\n\n")
x_new=[1,0,]

#Posicion, Equipo, Cantidad de 90 minutos jugados,competencia o liga (veamos de quitar al arquero de posicion)


#DF=defensores, MF=mediocampo, GK=arquero, FW=delanteros
#Vamos a quitar la posicion GK

#Vamos a utilizar anova para ver si la predictora Posicion y Competencia o liga brindan información importante
#para predecir el modelo.

x=df[df["Pos"]!="GK"]
y=x["Gls"]
x_1=pd.get_dummies(x["Pos"],drop_first=True)*1
x_2=pd.get_dummies(x["Squad"],drop_first=True)*1
x_3=pd.get_dummies(x["Comp"],drop_first=True)*1
x_4=x["90s"]


X_5=sm.add_constant(x_1)
X_6=sm.add_constant(x_3)

modelo_5=sm.OLS(y,X_5).fit()
modelo_6=sm.OLS(y,X_6).fit()

#Vamos a utilizar anova

n=len(x)
X_an=np.ones((n, 1))
modelo_an=sm.OLS(y,X_an).fit()

print("El test anova con el modelo de las posciciones.")
print(anova_lm(modelo_an,modelo_5))
print("El test anova con el modelo de las ligas.")
print(anova_lm(modelo_an,modelo_6))

#Vemos que las posiciones nos dan relevantes para la prediccion
#Pero además vemos que las ligas no nos dan una relevancia en la cantidad de goles metidos.

#Al modelo que creamos anteriormente (Modelo 4) le vamos a agregar esta predictora(Posiciones)


x_1=pd.get_dummies(df_validos["Pos"],drop_first=True)*1

x_2=df_validos[["SoT","90s","MP"]]
x_7=pd.concat([x_2,df_validos["Pos"],y_val],axis=1)
X_7=sm.add_constant(pd.concat([x_2,x_1],axis=1))

modelo_7=sm.OLS(y_val,X_7).fit()

print("Modelo N°7, modelo 4 + posiciones\n")
print(modelo_7.summary())

#Comparando estos dos modelos con sus  R-squared adjusted vemos que el modelo N°7 es el que mejor ajusta.

plt.figure()

sns.scatterplot(x_7,x="SoT",y="Gls",hue="Pos")
plt.title("Scatter plot Tiros al arco vs Goles")
plt.show()

plt.figure()

sns.scatterplot(x_7,x="90s",y="Gls",hue="Pos")
plt.title("Scatter plot doble tiempos jugados vs Goles")
plt.show()

plt.figure()

sns.scatterplot(x_7,x="MP",y="Gls",hue="Pos")
plt.title("Scatter plot Tiros al arco vs Goles")
plt.show()



#Verificacion de supuestos

sm.qqplot(modelo_7.resid,line=)

