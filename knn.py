#digit recognition using K-nearest neighbors
#Yang Zhou

import random
import math
import numpy as np



#normalize data to [0,1]
def normalize_dataset(dataset):
    for line in dataset:
        maximum = max(line)
        for i in range(len(line)):
            line[i] = line[i] / maximum



#calculate the Euclidean distance between two points
def distance(x, xi):
    d = 0.0
    for i in range(len(x)):
        d += (float(x[i])-float(xi[i])) ** 2
    d = math.sqrt(d)
    return d



#make prediction using knn
def knn_predict(X_train,y_train,X_test,k_value):
    test_results = []
    for i in X_test:
        all_distance =[]

        for j in X_train:
            dj = distance(i, j)
            all_distance.append(dj)
        
        knn_index = np.argsort(all_distance)[0:k_value]

        all_numbers = list(np.zeros(10))

        for nn in knn_index:
            all_numbers[y_train[nn]] += 1
        
        test_results.append(all_numbers.index(max(all_numbers))) 
    
    return test_results


#make prediction using weighted knn
def knn_predict_weighted(X_train,y_train,X_test,k_value):
    test_results = []
    for i in X_test:
        all_distance =[]

        for j in X_train:
            dj = distance(i, j)
            all_distance.append(dj)
        
        knn_index = np.argsort(all_distance)[0:k_value]

        #nearer neighbors are given higher weights
        weights = []
        for i in knn_index:
            d = all_distance[i]
            weights.append( 1 / ((d ** 2) + 1) )

        all_numbers = list(np.zeros(10))
        for i,nn in enumerate(knn_index):
            all_numbers[y_train[nn]] += (weights[i] / sum(weights))
        test_results.append(all_numbers.index(max(all_numbers))) 
    
    return test_results


#percentage accuracy
def accuracy(y_test,test_results):
    correct = 0
    for i,result in enumerate(test_results):
        if result == y_test[i]:
            correct += 1

    accuracy = float(correct)/len(y_test) *100  
    return accuracy



#reading files
#train set
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


#test set

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


#Q1
k = 1
while k <= 7:
    print("Q1, knn, k =" + str(k))
    results=knn_predict(X_train,y_train,X_test,k)

    print("accuracy:", accuracy(y_test,results))

    from sklearn.metrics import confusion_matrix
    cm_test=confusion_matrix(y_test, results)

    print("Confusion matrix for test data:")
    print(cm_test)
    k += 1
    
#Q2:
k = 1
while k <= 7:
    print("Q2, weighted knn, k =" + str(k))
    results=knn_predict_weighted(X_train,y_train,X_test,k)

    print("accuracy:", accuracy(y_test,results))

    from sklearn.metrics import confusion_matrix
    cm_test=confusion_matrix(y_test, results)

    print("Confusion matrix for test data:")
    print(cm_test)
    k += 1


#Q3:

k = len(X_train)
print("Q3, weighted knn, k =" + str(k))
results=knn_predict_weighted(X_train,y_train,X_test,k)

print("accuracy:", accuracy(y_test,results))

from sklearn.metrics import confusion_matrix
cm_test=confusion_matrix(y_test, results)

print("Confusion matrix for test data:")
print(cm_test)


