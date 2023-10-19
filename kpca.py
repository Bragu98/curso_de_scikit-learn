import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart_bde64b4c-2d72-4cd3-a964-62ee94855f5b.csv')

    #print(dt_heart.head(3))

    #Dividimos nuesto Data set en features y target 
    dt_features = dt_heart.drop(['target'], axis = 1 )
    dt_target = dt_heart['target']

    # para utilizar PCA necesitamos primero normalizar los datos con alguna funcion 
    dt_features = StandardScaler().fit_transform(dt_features)


    #ahora haremos la distribución de variables 
        #test_size sirve para determonar el tamaño del conjunto de entrnamiento que se quiere utilizar para posteiormente hacer pruebas (ejemplo 30% del data set)
        #random_state sirve para añadir replicabilidad para nuestro experimento
    X_train, X_test, Y_train, Y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    """
        KERNELS: Un Kernel es una función matemática
        que toma mediciones que se comportan
        de manera no lineal y las proyecta en un
        espacio dimensional más grande donde
        sean linealmente separables.
    """

    #declaramos nuestra variable kernel
        #podemos seleccionar la implementacio kernel que deseemos segun nuestras necesidades
    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(X_train)

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs')

    logistic.fit(dt_train, Y_train)
    print("SCORE KPCA: ", logistic.score(dt_test, Y_test))

    """
        Aplicar Kernels es algo demasiado facil, lo dificil es poder identificar cuando es necesario aplicarlos y cuando no
    """