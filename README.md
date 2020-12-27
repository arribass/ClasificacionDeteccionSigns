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

>>>La precision es 78.1
```
```
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(Xtrain,ytrain)

ypred = clf.predict(Xtest)
accuracyMNB = round(metrics.accuracy_score(ypred,ytest)*100,2)
print('La precision es {}'.format(accuracyMNB))

>>>La precision es 77.31
```

### SVMs
### Regresion logistica

## Con preprocesamiento

### Bayes

### SVMs

### Regresion logistica
## Estrategia deteccion:

Algoritmo de ventana deslizante. Usaremos las imagenes de dataset/images
