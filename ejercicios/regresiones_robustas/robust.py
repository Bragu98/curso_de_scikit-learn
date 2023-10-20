
"""
    IDENTIFICACIÓN DE DATOS ATIPICOS:

    ¿COMO IDENTIFICARLOS?
    A través de métodos estadísticos:
    1. Z – Score: Mide la distancia (en desviaciones estándar) de un punto dado a la media.
    2. Técnicas de clustering como DBSCAN.
    3. Si q< Q1 -1.5∗IQR ó q> Q3+1.5∗ IQR (ESTE SE PUEDE VER DE FORMA GRAFICA POR MEDIO DE LOS BOXPLOT)
"""

"""
    REGRESIONES ROBUSTAS:

    Sci-kit learn nos ofrece algunos modelos
    específicos para abordar el problema
    de los valores atípicos:
    1. RANSAC
    2. Huber Regressor
"""

"""
    RANSAC (Random Sample Consensus)

    Usamos una muestra aleatoria sobre el conjunto
    de datos que tenemos, buscamos la muestra que
    más datos “buenos” logre incluir.
    El modelo asume que los “malos valores” no tienen
    patrones específicos.
"""

"""
    HUBER REGGRESOR

    No ignora los valores atípicos, disminuye su
    influencia en el modelo.
    Los datos son tratados como atípicos si
    el error absoluto de nuestra pérdida está por
    encima de un umbral llamado epsilon.
    Se ha demostrado que un valor de epsilon = 1.35
    logra un 95% de eficiencia estadística.
"""

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import warnings


from sklearn.linear_model import (
    RANSACRegressor,
    HuberRegressor
)
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == ("__main__"):
    
    dataset= pd.read_csv ("./../../data/felicidad_corrupt.csv")
    #print(dataset.head(3))

    X= dataset.drop(["country", "score"], axis= 1)
    Y= dataset[["score"]]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

    #A continuación haremos una forma de ralizar varios estimadores al mismo tiempo, sin necesidad de repetir tanto codigo

    estimadores = {
        'SVR' : SVR(gamma= 'auto', C=1.0, epsilon=0.1),
        #Ransac no es un estimador, es una meta estimador, lo cual significa que me permite trabajar con diferentes estimadores definidos como si fuera un para metro Ejem: RANSACRegressor(SVR())
            #si dejamos a ransac por defecto, nos ejecuta una regresión lineal 
        'RANSAC' : RANSACRegressor(),
        # recordemos que en huber el parametro epsilon nos sirve para configurar que tantos datos van a ser considerados como atipocos entre mas o menos sea este valor 
            #sin embargo el valor recomendado y que viene por defecto es 1.35
        'HUBER' : HuberRegressor(epsilon= 1.35)
    }

    for name, estimador in estimadores.items():
        estimador.fit(X_train, Y_train)
        predictions = estimador.predict(X_test)
        warnings.simplefilter("ignore")

        print("="*32)
        print(name)
        print("MSE: ", mean_squared_error(Y_test, predictions))
        plt.ylabel('Predicted Score')
        plt.xlabel('Real Score')
        plt.title('Predicted VS Real')
        plt.scatter(Y_test, predictions)
        plt.plot(predictions, predictions,'r--')
        plt.show()

