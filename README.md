# Traffic sign detector and classifier

En este proyecto vamos a elaborar mediante lo aprendido en Vision Artificial y Aprendizaje Formal un programa que sea capaz de detectar y clasificar 43 tipos de señales de tráfico diferentes.

Las señales que soporta el clasificador son las siguientes:

![alt text](resources/signs.png)

## Estrategia clasificación:

En la carpeta [csvs](csvs/) podemos encontrar 43 csvs que contienen los datos de las imagenes de [Dataset_traffic_sign](Dataset_traffic_sign/) procesadas por el algoritmo HOG de opencv. Mas informacion [aqui](https://www.learnopencv.com/histogram-of-oriented-gradients/)

Utilizamos este algoritmo para crear los datos de cada clase y posteriormente crear otros 2 conjuntos de datos [train](train_img_features.npy) y [test](test_img_features.npy) para entrenar y probar clasificadores.

Para clasificar las señales usaremos:
- SVMs (OVO&OVA)
- Naive Bayes
- Regresion logistica multiclase

Todos ellos de las librerias de [sklearn](https://scikit-learn.org/stable/)

A partir de cada clasificador hemos conseguido los siguientes resultados:

## Sin preprocesamiento

### Bayes

Hemos entrenado 2 modelos de Naive Bayes: gaussiano y multiclase. Los resultados han sido los siguientes:
```
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(Xtrain, ytrain)

ypred = clf.predict(Xtest)
accuracyGaussianNB = round(metrics.accuracy_score(ypred,ytest)*100,2)
print('La precision es {}'.format(accuracyGaussianNB))

>>>La precision es 74.93
```
```
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(Xtrain,ytrain)

ypred = clf.predict(Xtest)
accuracyMNB = round(metrics.accuracy_score(ypred,ytest)*100,2)
print('La precision es {}'.format(accuracyMNB))

>>>La precision es 78.89
```

Apenas hay diferencia de precision entre ambos clasificadores por lo que ambos nos servirian igualmente.
### SVMs
```
from sklearn import svm
clf_SVM = svm.SVC()
clf_SVM.fit(Xtrain, ytrain)

ypred = clf_SVM.predict(Xtest)
accuracySVM = round(metrics.accuracy_score(ypred,ytest)*100,2)
print('La precision es {}'.format(accuracySVM))

>>>La precision es 87.34
```
En el caso de las SVMs hemos probado tanto OVO como OVA y hemos obtenido los mismos resultados. Hay mejora con respecto a Naive Bayes pero aun se puede mejorar mas.
### Regresion logistica
```
from sklearn.linear_model import LogisticRegression
clfLogRegr = LogisticRegression(random_state=0,max_iter=300).fit(Xtrain, ytrain)

ypred = clfLogRegr.predict(Xtest)
accuracyLogReg = round(metrics.accuracy_score(ypred,ytest)*100,2)
print('La precision es {}'.format(accuracyLogReg))

>>>La precision es 94.46
```
En regresión logistica es donde hemos obtenido el mayor porcentaje de acierto en la clasificacion por lo que nos quedaremos con este clasificador para las demas partes del trabajo.

Es importante resaltar que estas precisiones depende también de como se hayan mezclado los conjuntos de train y test. En nuestro caso hemos utilizado una semilla(2020) para fijar los resultados pero podrian variar. En cualquier caso hemos realizado pruebas y la regresión logistica sigue dando los mejores resultados.
## Con preprocesamiento

Antes de ver los resultados con preprocesamiento vamos a mostrar cual es el procedimiento que hemos seguido.


### Bayes

### SVMs

### Regresion logistica
## Estrategia deteccion:

Algoritmo de ventana deslizante. Usaremos las imagenes de dataset/images
