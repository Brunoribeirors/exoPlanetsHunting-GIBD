# exoPlanetsHunting-GIBD
#Projeto final da Disciplina de Gerência de Infraestrutura de Big Data

    Aluno: Bruno Martinez Ribeiro

Este trabalho teve como objetivo desenvolver uma análise dados utilizando Spark, acessando arquivos em ambiente Hadoop. Os dados análisados são observações realizadas pelo telescópio espacial Kepler, que capturou a luminosidade de estrelas em um determinado tempo a fim de analisar as variações de luminosidade e identificar possíveis exoplanetas orbitantes.
Dados obitidos do kaggle: https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data/

As ferramentas necessárias para executar a tarefa são:

    Hadoop 3.2.1
    Spark 3.0.1
    Python 3.6
    Jupyter

Configuração do Ambiente Hadoop

Esse trabalho utiliza algumas variáveis de ambiente do hadoop e spark, contidas no script abaixo:

source hadoop_vars.sh

Inicialização dos serviços namenode e datanode do hadoop, e resourcemanager e nodemanager do yarn, rodando o script abaixo:

. script.sh

Variáveis de ambiente para inicialização do Jupyter junto com o PySpark:

. jupyter_pyspark.sh

Ingestão do dado no HDFS:
Após realizar o download dos arquivos, executar os comando abaixo para ingestão de dados no HDFS (alterar o caminha da pasta de origem, caso tenha baixando em outra pasta)

hdfs dfs -put exoTrain exoTrain/exoTrain.csv
hdfs dfs -put exoTest exoTest/exoTest.csv

Análise de dados :
è possível executar o script .py no diretamente no pyspark ou copiar e colar os comando do script e executar no console do pyspark.
Também é possível carregar o notebook .ynp no Jupyter e executar o código
