#digit recognition using Neural Network
#Yang Zhou

import random
import copy
from math import exp
from sklearn.metrics import confusion_matrix



def normalize_dataset(dataset):
    for line in dataset:
        maximum = max(line)
        for i in range(len(line)):
            line[i] = line[i] / maximum





def get_accuracy(y_predict,y_target):
    accu = 0
    for i in range(len(y_predict)):
        if int(y_predict[i]) == int(y_target[i]):
            accu += 1.0
    accu = accu / len((y_predict))
    return accu

def get_results(output):
    return output.index(max(output))
    
    

def initialize_weights(n_inputs, n_neurons, n_outputs):
    weights = list()
    hidden_layer = [[random.uniform(-1,1) for i in range(n_inputs)] for i in range(n_neurons )]
    weights.append(hidden_layer)
    output_layer = [[random.uniform(-1,1) for i in range(n_neurons + 1)] for i in range(n_outputs)]
    weights.append(output_layer)

    return weights



def activate(weights, inputs):
    activation = 0
    for i in range(len(weights)):
        activation += weights[i] * inputs[i]
    return activation


# Forward propagate 
def forward_propagate(weights, row):
    inputs = row
    output = [1.0]
    output2 = []
    for neuron in range(n_hidden_neurons):
        w = weights[0][neuron]
        activation = activate(w, inputs)
        out = 1.0 / (1.0 + exp(-activation))
        output.append(out)
    #next layer
    for neuron2 in range(n_outputs):
        w = weights[1][neuron2]
        activation = activate(w, output)
        out = 1.0 / (1.0 + exp(-activation))
        output2.append(out)

    return output,output2




# Backpropagate
def backward_propagate(weights, input_row, output, output2,expected):
    #get gradient for output layer
    error_k = []
    
    for k,out in enumerate(output2):
        error = out * (1.0 - out) * (out - expected[k])
        error_k.append(error)

    #get gradient for hidden layer
    error_j = []
    for j,out in enumerate(output[1:]): #exclude bias unit
            error = out * ( 1.0 - out) * sum_jk_error(j,error_k,weights)
            error_j.append(error)


    ####update weights
    for k,row in enumerate(weights[1]):
        for j,cell in enumerate(row):
            delta = l_rate * error_k[k] * output[j]
            row[j] = row[j] - delta
    
    for j,row in enumerate(weights[0]):
        for i,cell in enumerate(row):
            delta = l_rate * error_j[j] * input_row[i]
            row[i] = row[i] - delta
        
    
    
def sum_jk_error(j,error_k,weights):
    sum_jk_error = 0
    for k in range(n_outputs):
        sum_jk_error += weights[1][k][j+1] * error_k[k]
    return sum_jk_error
        



# Train a network
def train_network(init_weights,train, l_rate, n_outputs,desired_accuracy):
    weights = copy.deepcopy(init_weights)
    y_predict = [0] * len(train)
    sum_error = 9999
    iteration = 0
    while get_accuracy(y_predict,y_train) < desired_accuracy:
        iteration += 1
        sum_error = 0
        for i,row in enumerate(train):
            output,output2 = forward_propagate(weights, row)
            y_predict[i] = get_results(output2)
            expected = [0.0 for i in range(n_outputs)]
            expected[y_train[i]] = 1.0
            sum_error += sum([(expected[i]-output2[i])**2 for i in range(len(expected))])
            backward_propagate(weights, row, output, output2, expected)
            
        if iteration % 10 == 0:
            print("#"+str(iteration))
            print(sum_error / len(train) / 2)  #MSE
            print (get_accuracy(y_predict,y_train))
            
    print("result on training set:")
    print("#"+str(iteration))
    print(sum_error / len(train) / 2)  #MSE
    print (get_accuracy(y_predict,y_train))  #accuracy
    
    return weights,y_predict




def predict(test,weights):
    y_predict = [0] * len(test)
    sum_error = 0
    for i,row in enumerate(test):
            output,output2 = forward_propagate(weights, row)
            y_predict[i] = get_results(output2)
            expected = [0.0 for i in range(n_outputs)]
            expected[y_test[i]] = 1.0
            sum_error += sum([(expected[i]-output2[i])**2 for i in range(len(expected))])
    return y_predict,sum_error


#reading files
random.seed(12345)

file=open('train.txt').readlines()
data=[]
target = []

random.shuffle(file)

for row in file:
    row = [float(x) for x in row.split(" ")]
    data.append(row[1:])
    target.append(int(row[0]))

X_train = data
y_train = target


normalize_dataset(X_train)

for i,row in enumerate(X_train):
    X_train[i] = [1.0] + row    #adding bias unit
    



file=open('test.txt').readlines()
test_data=[]
test_target = []
for row in file:
    row = [float(x) for x in row.split(" ")]
    test_data.append(row[1:])
    test_target.append(int(row[0]))

X_test = test_data
y_test = test_target

normalize_dataset(X_test)

for i,row in enumerate(X_test):
    X_test[i] = [1.0] + row    #adding bias unit




n_outputs = 10
l_rate= 0.1
n_hidden_neurons = 25
desired_accuracy = 0.98
init_weights = initialize_weights(len(X_train[0]), n_hidden_neurons, n_outputs)

weights_new,y_predict = train_network(init_weights, X_train, l_rate, n_outputs,desired_accuracy)

results,sum_error=predict(X_test,weights_new)
print ("Accuracy on test data: " +str(get_accuracy(results,y_test)))
print ("MSE on test data:" + str(sum_error / len(X_test) / 2 ))


#confusion matrix
cm_test=confusion_matrix(y_test, results)

print("Confusion matrix for test data:")
print(cm_test)


