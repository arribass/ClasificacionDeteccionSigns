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
### Algoritmo HOG

A partir de cada clasificador hemos conseguido los siguientes resultados:



Hemos probado a definir un conjunto de clasificadores y entrenarlos para quedarnos con el mejor resultado. El codigo es el siguiente:
Para ello tambien hemos usado GridSearchCV para obtener el mejor ajuste de los paremtros de los clasificadores

```
nombresDatasets = ['non_processed','processed']
datasets = {}

for nombre in nombresDatasets:
    pathTrain = 'train_img_features_'+nombre+'.npy'
    pathTest = 'test_img_features_'+nombre+'.npy'
    
    dataTrain = np.load(pathTrain)
    Xtrain = dataTrain[:,:-1]
    ytrain = dataTrain[:,-1]
    
    dataTest = np.load(pathTest)
    Xtest = dataTest[:,:-1]
    ytest = dataTest[:,-1]
    
    datasets[nombre] = (Xtrain,ytrain,Xtest,ytest)

#Definimos un conjunto de clasificadores
classifiers = [
    SVC(),
    LogisticRegression(random_state=0,max_iter=400),
    GaussianNB(),
    DecisionTreeClassifier(),
    MLPClassifier()]

clf_names = ['SVM', 'Regr Logistica', 'NB Gaussiano','Decision Tree','Red Neuronal']

score_list = []
time_list = []
aux_params = []
Best_score = np.NINF

#Definimos un diccionario con diccionarios para los parametergrids de GridSearchCV
parameters_dict = {'SVM':{'kernel':('linear', 'rbf'), 'C':[1, 10],'gamma':[0.1,0.001]},
                    'Regr Logistica':{'C': [0.01, 1, 10, 1000] },
                    'NB Gaussiano':{'var_smoothing': np.logspace(0,-9, num=10)},
                    'Decision Tree':{'min_samples_split': [2, 3, 4],
                                     'criterion': ['gini', 'entropy']},
                    'Red Neuronal':{'activation': ['tanh', 'relu'],
                                    'solver': ['sgd', 'adam']}
                   }
#Probamos los clasificadores
for i,(clf_aux,clf_name) in enumerate(zip(classifiers,clf_names)):
    for j,dataset in enumerate(nombresDatasets):
        #Clonamos el clasificador ya que lo vamos a usar 2 veces
        clf = clone(clf_aux)
        
        #Extraemos los datos de train y test
        print('Entrenando {} con {}'.format(dataset,clf_name))
        (Xtrain,ytrain,Xtest,ytest) = datasets[dataset]
        
        #Calculamos los parametros optimos con GridSearchCV
        t0 = time.time()
        clf = GridSearchCV(clf,parameters_dict[clf_name],cv = 2,n_jobs=-1)
        clf.fit(Xtrain,ytrain)
        t1 = time.time()
        time_list.append(round(t1-t0),2)
        
        #Calculamos la precision, la guardamos y vemos si hemos mejorado
        score = round(clf.score(Xtest,ytest)*100,2)
        score_list.append(score)
        aux_params.append(clf.best_params_)

        if score >= Best_score:
            Best_score = score
            Best_dataset = dataset
            Best_clf = clf
            nBest_clf = i*2 +j
        print('Time elapsed on fit: {}\n'.format(round(t1-t0,2)))
print('Total time elapsed {}'.format(round(np.sum(time_list),2)))
```
Importante resaltar tambien el parametro cv que hemos tenido que ajustar 'cv = 2' ya que para determinadas clases no habia suficientes ejemplos y no se podía realizar la validación cruzada.

Estos han sido los resultados tanto de precision como de tiempo de entrenamiento de los clasificadores

![alt text](resources/resultados.png)
![alt text](resources/resultados2.png)

Hemos tomado mejor clasificador finalmente MLPClassifier con los siguientes parametros
## Estrategia deteccion:

A partir de las imagenes en [dataset_images](dataset/images) hemos creado otro dataset: [dataset_cropped_images](dataset/images) con imagenes de tamaño 100x100 para entrenar otro clasificador que nos determine si dicha imagen es o no una señal

Posteriormente con el clasificador ya entrenado vamos a aplicar un algoritmo de ventana deslizante. Usaremos las imagenes completas de [dataset_images](dataset/images)


## Deteccion y clasificacion 