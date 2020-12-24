# Importando módulos utilizados
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

# Carregando arquivos csv
exoTrain = pd.read_csv('exoTrain/exoTrain.csv')
exoTest = pd.read_csv('exoTest/exoTest.csv')

# Dividindo dados em Label e Features
Y_train = np.reshape(np.array(exoTrain['LABEL']),(1,exoTrain.shape[0]))
X_train = np.transpose(np.array(exoTrain[exoTrain.columns[1:]]))
Y_test = np.reshape(np.array(exoTest['LABEL']),(1,exoTest.shape[0]))
X_test = np.transpose(np.array(exoTest[exoTest.columns[1:]]))

# Normalizando os dados
mean_train = np.reshape(np.mean(X_train,axis=0),(1,X_train.shape[1]))
std_train = np.reshape(np.std(X_train,axis=0),(1,X_train.shape[1]))
X_train = (X_train - mean_train)/std_train
mean_test = np.reshape(np.mean(X_test,axis=0),(1,X_test.shape[1]))
std_test = np.reshape(np.std(X_test,axis=0),(1,X_test.shape[1]))
X_test = (X_test - mean_test)/std_test

# Definindo estrutura da rede (3 camadas)
def defining_structure(X):
    n_i = X.shape[0]
    n_h1 = 12
    n_h2 = 12
    n_o = 1
    
    nodes ={
        "n_i":n_i,
        "n_h1":n_h1,
        "n_h2":n_h2,
        "n_o":n_o
    }
    
    return nodes

nodes = defining_structure(X_train)
print(nodes["n_h1"], nodes["n_h2"])

# Inicialização Randomica de cada camada
def random_initialization(X):
    np.random.seed(2)
    
    nodes = defining_structure(X)
    
    W1 = np.random.randn(nodes["n_h1"],nodes["n_i"])
    b1 = np.zeros((nodes["n_h1"],1))
    W2 = np.random.randn(nodes["n_h2"],nodes["n_h1"])
    b2 = np.zeros((nodes["n_h2"],1))
    W3 = np.random.randn(nodes["n_o"],nodes["n_h2"])
    b3 = np.zeros((nodes["n_o"],1))
    
    params = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2,
        "W3":W3,
        "b3":b3
    }
    
    return params

random_initialization(X_train)

# Função sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))

# Propagação para frente 
def forward_propagation(X, Y, parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = np.tanh(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = {
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2,
        "Z3":Z3,
        "A3":A3
    }
    
    return A3, cache

forward_propagation(X_train,Y_train, random_initialization(X_train))

# Custo computacional
def computing_cost(A3, Y):
    m = Y.shape[1]
    cost = -(np.dot(Y,np.transpose(np.log(A3))) + np.dot((1-Y),np.transpose(np.log(1-A3))))/m
    cost = np.squeeze(cost)
    return cost

# Propagação para trás
def backward_propagation(X,Y,parameters,cache):
    m = Y.shape[1]
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]
    Z3 = cache["Z3"]
    A3 = cache["A3"]
    
    dA3 = (-Y/A3) + (1-Y)/(1-A3)
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3,np.transpose(A2))/m
    db3 = np.sum(dZ3,axis=1,keepdims=True)/m
    
    dA2 = np.dot(np.transpose(W3),dZ3)
    dZ2 = dA2*(1-np.power(np.tanh(Z2),2))
    dW2 = np.dot(dZ2,np.transpose(A1))/m
    db2 = np.sum(dZ2,keepdims=True)/m
    
    dA1 = np.dot(W2,Z2)
    dZ1 = dA1*(1-np.power(np.tanh(Z1),2))
    dW1 = np.dot(dZ1,np.transpose(X))/m
    db1 = np.sum(dZ1,keepdims=True)/m
    
    grads = {
        "dW3":dW3,
        "dW2":dW2,
        "dW1":dW1,
        "db3":db3,
        "db2":db2,
        "db1":db1
    }
    
    return grads

# Funcção de atualização dos pesos e parâmetros
def update_weigths(learning_rate, parameters, grads):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    dW3 = grads["dW3"]
    dW2 = grads["dW2"]
    dW1 = grads["dW1"]
    db3 = grads["db3"]
    db2 = grads["db2"]
    db1 = grads["db1"]
    
    W3 = W3 - learning_rate*dW3
    W2 = W2 - learning_rate*dW2
    W1 = W1 - learning_rate*dW1
    b3 = b3 - learning_rate*db3
    b2 = b2 - learning_rate*db2
    b1 = b1 - learning_rate*db1
    
    params = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2,
        "W3":W3,
        "b3":b3,
    }
    
    return params

# Função modelo
def model(X,Y,learning_rate=0.03,num_iteration=3000):
    parameters = random_initialization(X)
    all_cost = list()
    
    for i in range(num_iteration):
        A3,cache = forward_propagation(X,Y,parameters)
        cost = computing_cost(A3,Y)
        all_cost.append(cost)
        print("Custo por iteração ",i," = ",cost,end='\r')
        
        grads = backward_propagation(X,Y,parameters,cache)
        parameters = update_weigths(learning_rate,parameters,grads)
        
    nn_model = {
        "grads":grads,
        "cache":cache,
        "parameters":parameters,
        "cost":all_cost
    }
    return nn_model

model_train = model(X_train, Y_train)

model_test = model(X_test, Y_test)

# Predição
train_prediction = np.squeeze(model_train["cache"]["A3"])
test_prediction = np.squeeze(model_test["cache"]["A3"])

print("Acurácia do conjunto Train =",(100 - np.mean(np.abs(train_prediction - Y_train))*100))
print("Acurácia do conjunto Test = ",(100 - np.mean(np.abs(test_prediction - Y_test))*100))


