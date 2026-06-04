import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class RegresionLineal:
    def __init__(self, x, y):
        # x = variables predictora/s
        # y = variable respuesta
        self.x = x
        self.y = y
        self.betas = None
        self.X=None
        self.result=None

    def ajustar_modelo(self):
      self.X=sm.add_constant(self.x)
      modelo=sm.OLS(self.y,self.X)
      self.result=modelo.fit()
      self.betas=self.result.params
      return self.betas

    def SE_conf(x, y, x_0):
      # x: variable predictora
      # y: variable respuesta
      # x_0: valor particular de la variable X donde interesa para el cual interesa estimar la esperanza de Y
      beta1_est = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
      beta0_est = np.mean(y) - beta1_est * np.mean(x)
      # estimación de sigma^2
      y_hat = beta0_est + beta1_est * x
      sigma2_est = np.sum((y - y_hat) ** 2) / (len(x) - 2)
      # varianza estimada de beta1_est
      var_beta1_est = sigma2_est / (sum((x - np.mean(x)) ** 2))
      # varianza estimada de beta0_est
      var_beta0_est = sigma2_est * np.sum(x ** 2) / (len(x)*sum((x - np.mean(x)) ** 2))
      # covarianza de beta0_est y beta1_est
      cov_01 = - np.mean(x) * sigma2_est / sum((x - np.mean(x)) ** 2)
      SE2_est = var_beta0_est + (x_0 ** 2) * var_beta1_est + 2 * x_0 * cov_01
      return float(np.sqrt(SE2_est))



    def graficar_qqplot(self,lista,gra=True):
        if self.betas is None:
          self.ajustar_modelo()

        n=len(lista)
        lista_ord=np.sort(lista)

        x_ord=np.arange(1,n+1)/(n+1)
        x_ord_s=(lista_ord-np.mean(lista))/np.std(lista)

        teorica=norm.ppf(x_ord)
        if gra:
          plt.figure()
          plt.scatter(x_ord_s,teorica)
          plt.show()
        return[x_ord_s,teorica]

    def obtener_estadisticas(self):
        if self.betas is None:
          self.ajustar_modelo()

        modelo=sm.OLS(self.y,self.X)
        result=modelo.fit()
        IC=result.conf_int()

        p_values=[result.p_values.iloc[0],result.p_values.iloc[1]]
        err_est=[result.bse.iloc[0],result.bse.iloc[1]]
        int_conf=[IC.iloc[0].values,IC.iloc[1].values]

        dic={
               "p_valor": p_values,
               "error_estandar": err_est,
               "IC_inf": [[int_conf[i][0]] for i in range(len(self.x.shape[1]))],
               "IC_sup": [[int_conf[i][1]] for i in range(len(self.x.shape[1]))]
           }
        return dic

    def errores(self,qqplot=True,res_vs_pred=True):
        if self.betas is None:
          self.ajustar_modelo()
        Y=np.dot(self.X,self.betas)
        r=[self.y[i]-Y[i] for i in range(len(Y))]
        if qqplot and res_vs_pred:
          qq=self.graficar_qqplot(r,False)
          _,axs=plt.subplots(1,2)
          axs[0].scatter(qq[1],qq[0])
          axs[0].plot([min(qq[1]),max(qq[1])],[min(qq[1]),max(qq[1])],c="r",linestyle="--")
          axs[0].set_title("Verificación Supuesto \n dist. normal")

          
          axs[1].scatter(Y,r)
          axs[1].plot([min(Y),max(Y)],[0,0],c="r",linestyle="--")
          axs[1].set_title("Verificación supuesto \n media 0")

          plt.show()

        
        return r


    def predecir_intervalos(self, x_new, nivel ,intervalo=None):
         if self.betas is None:
           self.ajustar_modelo()
           if isinstance(x_new,(float,int)):
            X_new= sm.add_constant(np.array([[1, x_new]]))
           else:
            X_new=sm.add_constant([x_new])
           prediccion=self.result.get_prediction(X_new)


         if intervalo=="confianza":
           return prediccion.conf_int(alpha=(1-nivel))
         if intervalo=="prediccion":
           return prediccion.conf_int(alpha=1-nivel,obs=1)

    def obtener_R2(self):
      if self.betas is None:
        self.ajustar_modelo()
      r_square=self.result.rscuared
      return r_square


class RegresionLinealSimple(RegresionLineal):
    def __init__(self, x, y):
        super().__init__(x, y)



    def predecir(self, x_0):
      if self.betas is None:
        super().ajustar_modelo()

      X_0 = sm.add_constant(x_0)
      prediccion=np.dot(X_0,self.betas)

      return prediccion

    def graficar_recta_ajustada(self):
        if self.betas is None:
          self.ajustar_modelo()
        plt.scatter(self.x, self.y, marker='o', c='blue', label='Datos', s=30)
        plt.plot(self.x, self.predecir(self.x), linestyle='--', color='black', label='Recta estimada')
        plt.grid(True, linestyle='--', alpha=0.7)


        plt.legend()
        plt.xlabel('experiencia')
        plt.ylabel('salario')
        plt.title('')
        plt.show()
    


class RegresionLinealMultiple(RegresionLineal):
    def __init__(self, x, y):
        super().__init__(x, y)

    def predecir(self,x_0):
      if self.betas is None:
        super().ajustar_modelo()

      X_0=sm.add_constant(x_0)
      prediccion=np.dot(X_0,self.betas)

      return prediccion

    def obtener_R2_ajustado(self):
      if self.betas is None:
        self.ajustar_modelo()
      r_square=self.result.rscuared_adj
      return r_square
