   

"""
        Otra tecnica que podemos utilizar pare el overfitting es la REGULARIZACIÓN:

        La regularización consiste en disminuir
        la complejidad del modelo a través de
        una penalización aplicada a sus variables más
        irrelevantes. Esto se realiza aplicando un poco de sesgo a nuestros datos

"""

"""
        Exiaten 3 tipos de regularización:

        ● L1 Lasso: Reducir la complejidad a través de la eliminación
        de features que no aportan demasiado al modelo.
        ● L2 Ridge: Reducir la complejidad disminuyendo el
        impacto de ciertos features a nuestro modelo.
        ● ElasticNet: Es una combinación de las dos anteriores.
"""



import pandas as pd 
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__" :
    dataset = pd.read_csv("./data/felicidad_b0b50c6d-41dd-4ea8-a4f0-92a8068d4d3e.csv")
    
    #print(dataset.describe())

    #pasamos a definir los features
     #se utiliza el doble [[]] para decirle a pandas que estamos operando sobre las columnas del dtset
    X= dataset[["gdp", "family", "lifexp", "freedom", "corruption", "generosity", "dystopia"]]
    Y= dataset[["score"]]
    #print(X.shape)
    #print(Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25)

    # Primero realizamos un modelo lineal para compararlo posteriormente con el de regularización 
    model_linear = LinearRegression().fit(X_train, Y_train)
    Y_predict_linear = model_linear.predict(X_test)

    #Vamos a utilizar el tipo de regulador Lasso
        #alpha nos sirve para determinar el grado de penalizacion que queremos aplicar a los features
    """
        Si hay overfitting lo mejor seria aumentar el valor de alpha, de lo contrario si hay underfitting disminuir el valor de alpha.
    """
    modelLasso = Lasso(alpha=0.02).fit(X_train, Y_train)
    Y_predict_lasso = modelLasso.predict(X_test)

    # Ahora utilizaremos el metodo Ridge 
    modelRidge = Ridge(alpha=1).fit(X_train, Y_train)
    Y_predict_ridge = modelRidge.predict(X_test)

    #Ahora visualizaremos las perdidas que hemos ejecutado y compararemos nuestros modelos 
    linear_loss= mean_squared_error(Y_test, Y_predict_linear)
    print("Linear Loss: ", linear_loss)

    lasso_loss= mean_squared_error(Y_test, Y_predict_lasso)
    print("Lasso Loss: ", lasso_loss)

    ridge_loss= mean_squared_error(Y_test, Y_predict_ridge)
    print("Ridge Loss: ", ridge_loss )


    #Tambien podemos ver como afecta el coeficiente de nuestras regreciones 
        #El coeficiente es el valor por el cual multiplica los features para reducir su impacto 
    print("="*32)
    #En lasso el coeficiente puede llegar a ser reducido a 0 
    print("Coef LASSO")
    print(modelLasso.coef_)

    print("="*32)
    #en Ridge puede reducir su impacto a valores cercanos a 0 pero nunca a 0 absoluto 
    print("Coef RIDGE")
    print(modelRidge.coef_)


    #Ahora vamos a graficar cada uno de los modelos 
    residuals_l = np.subtract(Y_test, Y_predict_linear)
    residuals_r = np.subtract(Y_test, Y_predict_ridge)
    residuals_ls = np.subtract(Y_test, Y_predict_lasso.reshape((-1, 1)))
    plt.scatter(Y_predict_linear, residuals_l)
    plt.scatter(Y_predict_ridge, residuals_r)
    plt.scatter(Y_predict_lasso, residuals_ls)
    plt.axhline(y=0, color='r', linestyle='--')  # Agregar la línea en el valor cero
    plt.show()

    """
        Como se puede observar, el modelo que presenta menor perdida de datos y mayor precisión es el modelo lineal
    """

    #tambien podemos visualizar el SCORE

    print ("="*32)
    print("Score Lineal")
    print(model_linear.score(X_test, Y_test))
    print("Score Lasso")
    print(modelLasso.score(X_test, Y_test))
    print("Score Ridge")
    print(modelRidge.score(X_test, Y_test))

    """
        Es importante recordar que:
        
        - Ninguna de las dos es mejor que la otra para todos los casos.

        - Lasso envía algunos coeficientes a cero permitiendo así seleccionar variables significativas para el modelo.

        - Lasso funciona mejor si tenemos pocos predictores que influyen sobre el modelo.

        - Ridge funciona mejor si es el caso contrario y tenemos una gran cantidad.

        Para aplicarlos y decidir cuál es el mejor en la práctica, podemos probar usando alguna técnica como cross-validation iterativamente. o bien, podemos combinarlos…
    """

    """
        Regularización ElasticNet

        Es común encontrarnos en la literatura con un camino intermedio llamado ElasticNet. Esta técnica consiste en combinar las dos penalizaciones anteriores en una sola función.   

        
        Donde tenemos ahora un parámetro adicional 𝛂 que tiene un rango de valores entre 0 y 1. Si 𝛂 = 0 , ElasticNet se comportará como Ridge, y si 𝛂 = 1 , se comportará como Lasso. Por lo tanto, nos brinda todo el espectro lineal de posibles combinaciones entre estos dos extremos.

        - Tenemos una forma de probar ambas L1 y L2 al tiempo sin perder información.

        - Supera las limitaciones individuales de ellas.

        - Si hace falta experiencia, o el conocimiento matemático de fondo, puede ser la opción preferente para probar la regularización.
    """

    """
        Para implementar esta técnica añadimos primero el algoritmo ubicado en el módulo linear_model.

        from sklearn.linear_model import ElasticNet

        Y luego simplemente lo inicializamos con el constructor ElasticNet() y entrenamos con la función fit().

        regr = ElasticNet(random_state=0)

        regr.fit(X, y)
    """