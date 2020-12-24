#export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre
export HADOOP_HOME=$HOME/hadoop
export FLUME_HOME=$PWD/apache-flume-1.9.0-bin
export SQOOP_HOME=$HOME/sqoop-1.4.7.bin__hadoop-2.6.0
export PIG_HOME=$HOME/pig-0.17.0
export HIVE_HOME=$PWD/apache-hive-3.1.2-bin
export ZEPPELIN_HOME=/home/hadoop/zeppelin-0.8.2-bin-all
export SPARK_HOME=$HOME/spark-3.0.1-bin-hadoop3.2
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HOME/hadoop/bin:$HOME/hadoop/sbin:$FLUME_HOME/bin:/$SQOOP_HOME/bin:$PIG_HOME/bin:$HIVE_HOME/bin:$ZEPPELIN_HOME/bin:$SPARK_HOME/bin

