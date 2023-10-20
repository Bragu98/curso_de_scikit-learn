import pandas as pd 
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import warnings


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    
    dt_heart=pd.read_csv("./../../data/heart_bde64b4c-2d72-4cd3-a964-62ee94855f5b.csv")
    print(dt_heart["target"].describe())

    X= dt_heart.drop(['target'], axis= 1)
    Y= dt_heart["target"]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

    #vamos a utilizar un clasificador poco adecuado o de bajo acierto
    knn_class= KNeighborsClassifier(). fit(X_train, Y_train)
    knn_pred= knn_class.predict(X_test)

    print("="*50)
    print(accuracy_score(knn_pred, Y_test))
    warnings.simplefilter("ignore")

    #bagging es un meta clasificador, aqui colocaremos como parametro un clasificador poco confiable, n esrimators es el numero de estimadores por el cual queremos que pase 
    bag_class= BaggingClassifier(base_estimator= KNeighborsClassifier(), n_estimators= 50).fit(X_train,Y_train)
    bag_pred= bag_class.predict(X_test)

    print("="*50)
    print(accuracy_score(bag_pred, Y_test))

    #TAREA: Realizar el proceso con otros modelos de clasificación y hacer la compración de los resultados
