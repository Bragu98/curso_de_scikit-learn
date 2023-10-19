import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
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

    print(X_train.shape)
    print(Y_train.shape)

    #Ahora vamos a llamar y configurar nuesto algoritmo PCA
        #n_components = min(n_muentras, n_features)
    pca = PCA(n_components=3)
    pca.fit(X_train)    #para utilizar pca despues de ser configurada, vamos a utilizar la función fit, aplicada a los datos de entrenamiento, esto para que el pcs se ajuste a los datos que tenemos 

    #ahora vamos a utilizar la funcion de inclemental PCA
        # debido a que ipca nos sirve para entrenan modelos en computadores con pocos recursos, haciendo entrenamientos por bloques y no todos de una, batch_size nos sirve para definir el tamaño de dichos bloques
    ipca= IncrementalPCA(n_components=3, batch_size= 10)
    ipca.fit(X_train)

    #una vez ajustados los datos, procedemos a medir la varianza de los componentes que extrae 
        #generamos una representación grafica, graficamos el eje x llamando los numeros entre 0 y la cantidad de componentes que genero pca, esto con el comendo pca.explained_variance_
        #clasificamos el eje Y el valor de la importancia de cada componente del eje x con el comando explained_variance_ratio_ (esto para identificar los componentes importantes para nuestro modelo)
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    #plt.show()

    """
        La grafica nos muestra que los componentes 1 y 2 son los unicos que generan cierta relevancia para nuestros datos, miestras que el componente 3 nos refleja 
        una una relevancia casi nula
    """

    #procedemos a configurar nuestra variable logistic
        # solver='lbgfgs' es un parametro por defecto que nos ayudara a evitar algunos errores en el futuro
    logistic = LogisticRegression(solver='lbfgs')

    # despues de tener configurada nuestra LogisticRegression, ya podemos enviar nuestro entrenamiento - prueba con pca y ipca
    # para que nuestra prueba sea exitosa, debemos aplicar pca tanto a nuestro entrenamiento, como a nuestra prueba 
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train,Y_train)
    # para medir la efectividad del modelo, tenemos que medir algunas de las metricas, en este caso mediremos uno sensillo como ejemplo
    print("SCORE PCA: ", logistic.score(dt_test, Y_test))

    #ahora haremos lo mismo pero con ipca
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train,Y_train)
    print("SCORE IPCA: ", logistic.score(dt_test, Y_test))

    """
        SCORE PCA:  0.7857142857142857
        SCORE IPCA:  0.8051948051948052

        Con esto podemos ver que utilizando solo 3 features proporcionados por pca, obtenemos un porcentaje de acierto muy alto sin consumir tantos recursos
    """