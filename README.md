# Traffic sign detector and classifier

En este proyecto vamos a elaborar mediante lo aprendido en Vision Artificial y Aprendizaje Formal un programa que sea capaz de detectar y clasificar 43 tipos de señales de tráfico diferentes.

Las señales que soporta el clasificador son las siguientes:

![alt text](resources/signs.png)

## Estrategia clasificación:

En la carpeta [csvs](csvs/) podemos encontrar 43 csvs que contienen los datos de las imagenes de [Dataset_traffic_sign](Dataset_traffic_sign/) procesadas por el algoritmo HOG de opencv. Mas informacion [aqui](https://www.learnopencv.com/histogram-of-oriented-gradients/)

Utilizamos este algoritmo para crear los datos de cada clase y posteriormente crear otros 2 conjuntos de datos [train](train_img_features.npy) y [test](test_img_features.npy) para entrenar y probar clasificadores.

Para clasificar las señales usaremos:
- SVMs
- Naive Bayes
- Regresion logistica multiclase
- Decision tree
- Red Neuronal (MLPClassifier)

Todos ellos de las librerias de [sklearn](https://scikit-learn.org/stable/)
### HISTOGRAM ORIENTED GRADIENT (HOG)

El algoritmo HOG es un metodo de extraccion de caracteristicas que nos permite transformar los pixeles por gradientes que nos indican hacia donde se orientan los cambios de intensidad de dichos pixeles. Esto es muy util porque nos permite obtener caracteristicas muy representativas de una imagen y que nos serviran para comparar con otras similares, ya sean caras, señales de tráfico, etc...

Hemos definido un conjunto de clasificadores y entrenarlos para quedarnos con el mejor resultado:
```
classifiers = [ SVC(),
                LogisticRegression(random_state=0,max_iter=400),
                GaussianNB(),
                DecisionTreeClassifier(),
                MLPClassifier()]

clf_names = ['SVM', 'Regr Logistica', 'NB Gaussiano','Decision Tree','Red Neuronal']
```
Para ello tambien hemos usado GridSearchCV para obtener el mejor ajuste de los paremtros de los clasificadores

```
parameters_dict = {'SVM':{'kernel':('linear', 'rbf'), 'C':[1, 10],'gamma':[0.1,0.001]},
                   'Regr Logistica':{'C': [0.01, 1, 10, 1000] },
                   'NB Gaussiano':{'var_smoothing': np.logspace(0,-9, num=10)},
                   'Decision Tree':{'min_samples_split': [2, 3, 4],
                                     'criterion': ['gini', 'entropy']},
                   'Red Neuronal':{'activation': ['tanh', 'relu'],
                                    'solver': ['sgd', 'adam']}
                  }
```
Importante resaltar tambien el parametro cv que hemos tenido que ajustar 'cv = 2' ya que para determinadas clases no habia suficientes ejemplos y no se podía realizar la validación cruzada y el parametro n_jobs que hace trabajar al procesador en paralelo para poder entrenar mas rápido.
```
clf = GridSearchCV(clf,parameters_dict[clf_name],cv = 2,n_jobs=-1)
```

Estos han sido los resultados tanto de precision como de tiempo de entrenamiento de los clasificadores

![alt text](resources/resultados.png)
![alt text](resources/resultados2.png)

Hemos tomado mejor clasificador finalmente Regresión Logistica ya que hemos obtenido una precisión del 94.2% y tiene un tiempo de entrenamiento razonable. Estos han sido los parametros utilizados
```

```
Lo guardamos en nuestro repositorio usando [pickle](https://docs.python.org/3/library/pickle.html) de este esta manera tenemos el objeto estimador guardado en disco sin necesidad de tener que volver a crear los datasets,entrenar y calcular el mejor.
```

```
## Estrategia deteccion:

A partir de las imagenes en [dataset_images](dataset/images) hemos creado otro dataset: [dataset_cropped_images](dataset/images) con imagenes de tamaño 100x100 para entrenar otro clasificador que nos determine si dicha imagen es o no una señal

Posteriormente con el clasificador ya entrenado vamos a aplicar un algoritmo de ventana deslizante. Usaremos las imagenes completas de [dataset_images](dataset/images). 

De primeras entrenamos una SVM que nos ofrecia bastantes buenos resultados y nos detectaba las señales con precisión pero nos dimos cuenta de que nos podía detectar una misma señal multiples veces al aplicar la ventana deslizante. Por ello decidimos aplicar Non-Maximum Supression para eliminar las detecciones multiples, para ello cambiamos el modelo de detección por un clasificador probabilístico que nos permitiera quedarnos con las ventanas con mayor probabilidad de su clase señal. Finalmente nos hemos quedado con un MLP-Classifier de Sklearn.




## Deteccion y clasificacion 

Una vez tenemos un modelo q nos clasifique los distintos tipos de señales y hemos diseñado un algoritmo que sea capaz de detectar en una imagen donde se encuentra una señal, entonces podemos combinar los dos clasificadores para detectar y clasificar dicha señal.