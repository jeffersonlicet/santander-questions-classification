# Santander Questions Classification 2020 | 🏅 TOP 3 - 0.86639

## Solution:

![](https://camo.githubusercontent.com/c814494ff12e6337a262bf025f01d3d8eefb3725/68747470733a2f2f692e696d6775722e636f6d2f56744b6437704b2e706e67)

Indicaciones para entrenar los modelos:

Se adjuntan 4 archivos:
* LSTM_GRU_0_86.ipynb Notebook con todo el código para entrenar los modelos basados en LSTM y GRU
* BERT_0_86.ipynb Notebook con todo el código para entrenar el modelo basado en BERT
* Informe.ipynb Notebook con el informe y el análisis de datos.
* assemble.py Script que ensambla las predicciones de los 3 modelo y genera un archivo listo para ser enviado a la competencia.

# 1: Entrenar los modelos:

Correr los notebooks preferiblemente en paralelo y utilizando Google Colaboratory.
  * LSTM_GRU_0_86.ipynb con GPU Activado
  * BERT_0_86.ipynb con TPU Activado

# 2: Descargar los archivos
Una vez finalizado el entrenamiento, que toma algo más de una hora, ambos notebooks van a intentar
descargar archivos, en caso de que no posean permisos para descargarlos por parte del browser puede
intentar descargarlos manualmente.

# 3: Ensamblar la solución
Archivos que deben estar descargados y en el mismo directorio:
* test_ids.npy Contiene los ids de los casos de testing
* labels.npy Contiene los nombres de las clases mapeadas a indices
* bert.npy Contiene la distribución de probabilidad calculada usando BERT
* lstm.npy Contiene la distribución de probabilidad calculada usando LSTM
* gru.npy Contiene la distribución de probabilidad calculada usando GRU

Con los archivos en el mismo directorio que el archivo assemble.py que se encuentra
adjuntado realizar lo siguiete:

> pip install numpy

y luego 
> python assemble.py

Finalmente se va a generar un archivo llamado submission.csv con las predicciones correspondientes.


Gracias.
