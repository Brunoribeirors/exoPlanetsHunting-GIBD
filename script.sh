#source hadoop_vars.sh
hdfs --daemon stop namenode
hdfs --daemon stop datanode
yarn --daemon stop resourcemanager
yarn --daemon stop nodemanager
yarn --daemon stop timelineserver
mapred --daemon stop historyserver
zeppelin-daemon.sh stop
hdfs --daemon start namenode
hdfs --daemon start datanode
yarn --daemon start resourcemanager
yarn --daemon start nodemanager
yarn --daemon start timelineserver
mapred --daemon start historyserver
zeppelin-daemon.sh start
jps
