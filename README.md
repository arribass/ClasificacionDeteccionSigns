# Traffic sign detector and classifier

En este proyecto vamos a elaborar mediante lo aprendido en Vision Artificial y Aprendizaje Formal un programa que sea capaz de detectar y clasificar 43 tipos de señales de tráfico diferentes.

Las imagenes que soporta el clasificador son las siguientes:

![alt text](resources/signs.png)

## Estrategia clasificación:

Utilizamos el algoritmo HOG para extraer las caracteristicas de nuestra imagen y a partir de ahi entrenamos nuestro clasificador

Para clasificar las señales podemos usar diferentes estrategias:
- SVMs (OVO&OVA)
- Regresion lineal multiclase etc

## Estrategia deteccion:

Algoritmo de ventana deslizante. Usaremos las imagenes de dataset/images

En la carpeta [csvs](csvs/) podemos encontrar 43 csvs que contienen los datos de las imagenes de [Dataset_traffic_sign](Dataset_traffic_sign/) procesador por el algoritmo HOG de sklearn. Mas informacion [aqui](https://www.learnopencv.com/histogram-of-oriented-gradients/)