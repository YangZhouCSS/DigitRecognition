#digit recognition using Support Vector Machine
#Yang Zhou

import random
import math
import cvxopt
import numpy as np
import itertools




#normalize data to [0,1]
def normalize_dataset(dataset):
    for line in dataset:
        maximum = max(line)
        for i in range(len(line)):
            line[i] = line[i] / maximum




class SVMTrainer(object):
    def __init__(self, c, r):
        self._c = c
        self._r = r

    def train(self, X, y):   #train the model using the train set
        lagrange_multipliers = self._compute_multipliers(X, y)
        return self._construct_predictor(X, y, lagrange_multipliers)

    def _gram_matrix(self, X):   #compute the kernel matrix K
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))

        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = radial_basis_function_machine(x_i,x_j,self._r)
        return K

    def _construct_predictor(self, X, y, lagrange_multipliers):
        support_vector_indices =             lagrange_multipliers > 0.00001
            
        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        bias = np.mean(
            [y_k - SVMPredictor(
                bias=0.0,
                r=self._r,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])
        
        #bias = 0.0

        return SVMPredictor(
            bias=bias,
            r=self._r,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)

    def _compute_multipliers(self, X, y):
        #This function solves the dual form soft-margin of the svm optimization function
        n_samples, n_features = X.shape

        K = self._gram_matrix(X)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        return np.ravel(solution['x'])  #obatin lagrange_multipliers




class SVMPredictor(object):
    def __init__(self,
                 bias,
                 r,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._bias = bias
        self._r = r
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * radial_basis_function_machine(x_i,x,self._r)
        
        #return result
        return np.sign(result)#.item()
    



def radial_basis_function_machine(x1,x2,r):

    return (np.exp(np.linalg.norm(np.array(x1) - np.array(x2)) ** 2 * (-r)))


#get the accuracy
def get_accuracy (result,y_test):
    count_right = 0
    for i,r in enumerate(result):
        if r == y_test[i]:
            count_right += 1

    print("accuracy =", count_right / len(y_test))



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



print("Q3 classify 3 and 6")



X_train_Q3 = []
y_train_Q3 = []

for i, yt in enumerate(y_train):
    if yt == 3:
        X_train_Q3.append(X_train[i])
        y_train_Q3.append(1.0)
    if yt == 6:
        X_train_Q3.append(X_train[i])
        y_train_Q3.append(-1.0)

X_train_Q3 = np.array(X_train_Q3)
y_train_Q3 = np.array(y_train_Q3)
        
X_test_Q3 = []
y_test_Q3 = []

for i, yt in enumerate(y_test):
    if yt == 3:
        X_test_Q3.append(X_test[i])
        y_test_Q3.append(1.0)
    if yt == 6:
        X_test_Q3.append(X_test[i])
        y_test_Q3.append(-1.0)

X_test_Q3 = np.array(X_test_Q3)
y_test_Q3 = np.array(y_test_Q3)



trainer = SVMTrainer(c=100,r=0.05 )
predictor = trainer.train(np.array(X_train_Q3), np.array(y_train_Q3))



result = []
for row in X_test_Q3:
    rs=predictor.predict(row)
    result.append(rs)

get_accuracy(result,y_test_Q3)



print("Q5 decrease 50% features")
random.seed(12345)
deleteList = random.sample(range(784), round(784 * 0.50))
X_train_Q5_50 = np.delete(X_train_Q3, deleteList, axis=1)
y_train_Q5_50 = y_train_Q3

X_test_Q5_50 = np.delete(X_test_Q3, deleteList, axis=1)
y_test_Q5_50 = y_test_Q3

predictor = trainer.train(np.array(X_train_Q5_50), np.array(y_train_Q5_50))

result = []
for row in X_test_Q5_50:
    rs=predictor.predict(row)
    result.append(rs)

get_accuracy(result,y_test_Q5_50)



print("Q5 decrease 75% features")

deleteList = random.sample(range(784), round(784 * 0.75))
X_train_Q5_75 = np.delete(X_train_Q3, deleteList, axis=1)
y_train_Q5_75 = y_train_Q3

X_test_Q5_75 = np.delete(X_test_Q3, deleteList, axis=1)
y_test_Q5_75 = y_test_Q3

predictor = trainer.train(np.array(X_train_Q5_75), np.array(y_train_Q5_75))

result = []
for row in X_test_Q5_75:
    rs=predictor.predict(row)
    result.append(rs)

get_accuracy(result,y_test_Q5_75)



print("Q5 decrease 90% features")

deleteList = random.sample(range(784), round(784 * 0.90))
X_train_Q5_90 = np.delete(X_train_Q3, deleteList, axis=1)
y_train_Q5_90 = y_train_Q3

X_test_Q5_90 = np.delete(X_test_Q3, deleteList, axis=1)
y_test_Q5_90 = y_test_Q3

predictor = trainer.train(np.array(X_train_Q5_90), np.array(y_train_Q5_90))

result = []
for row in X_test_Q5_90:
    rs=predictor.predict(row)
    result.append(rs)

get_accuracy(result,y_test_Q5_90)


print("Q5 decrease 95% features")

deleteList = random.sample(range(784), round(784 * 0.95))
X_train_Q5_95 = np.delete(X_train_Q3, deleteList, axis=1)
y_train_Q5_95 = y_train_Q3

X_test_Q5_95 = np.delete(X_test_Q3, deleteList, axis=1)
y_test_Q5_95 = y_test_Q3

predictor = trainer.train(np.array(X_train_Q5_95), np.array(y_train_Q5_95))

result = []
for row in X_test_Q5_95:
    rs=predictor.predict(row)
    result.append(rs)

get_accuracy(result,y_test_Q5_95)



print("Q6 Apply SVD - 50%")
from sklearn.decomposition import PCA

X_all =np.concatenate((X_train_Q3, X_test_Q3), axis=0)


pca = PCA(n_components = round(784 * 0.50), svd_solver = 'arpack')
pca.fit(X_all)

X_all =  pca.transform(X_all)
X_train_Q6_50 = X_all[0:500]
X_test_Q6_50 = X_all[500:]


predictor = trainer.train(np.array(X_train_Q6_50), np.array(y_train_Q3))

result = []
for row in X_test_Q6_50:
    rs=predictor.predict(row)
    result.append(rs)

get_accuracy(result,y_test_Q3)



print("Q6 Apply SVD - 75%")

X_all =np.concatenate((X_train_Q3, X_test_Q3), axis=0)


pca = PCA(n_components = round(784 * 0.25), svd_solver = 'arpack')
pca.fit(X_all)

X_all =  pca.transform(X_all)
X_train_Q6_75 = X_all[0:500]
X_test_Q6_75 = X_all[500:]


predictor = trainer.train(np.array(X_train_Q6_75), np.array(y_train_Q3))

result = []
for row in X_test_Q6_75:
    rs=predictor.predict(row)
    result.append(rs)

get_accuracy(result,y_test_Q3)



print("Q6 Apply SVD - 90%")

X_all =np.concatenate((X_train_Q3, X_test_Q3), axis=0)


pca = PCA(n_components = round(784 * 0.10), svd_solver = 'arpack')
pca.fit(X_all)

X_all =  pca.transform(X_all)
X_train_Q6_90 = X_all[0:500]
X_test_Q6_90 = X_all[500:]


predictor = trainer.train(np.array(X_train_Q6_90), np.array(y_train_Q3))

result = []
for row in X_test_Q6_90:
    rs=predictor.predict(row)
    result.append(rs)

get_accuracy(result,y_test_Q3)



print("Q6 Apply SVD - 95%")

X_all =np.concatenate((X_train_Q3, X_test_Q3), axis=0)


pca = PCA(n_components = round(784 * 0.05), svd_solver = 'arpack')
pca.fit(X_all)

X_all =  pca.transform(X_all)
X_train_Q6_95 = X_all[0:500]
X_test_Q6_95 = X_all[500:]


predictor = trainer.train(np.array(X_train_Q6_95), np.array(y_train_Q3))

result = []
for row in X_test_Q6_95:
    rs=predictor.predict(row)
    result.append(rs)

get_accuracy(result,y_test_Q3)




print("Q7 reduce number of samples - 50%")

deleteList = random.sample(range(500), round(500 * 0.50))

X_train_Q7_50 = np.delete(X_train_Q3, deleteList, axis=0)
y_train_Q7_50 = np.delete(y_train_Q3, deleteList)

X_test_Q7_50 = np.delete(X_test_Q3, deleteList, axis=0)
y_test_Q7_50 = np.delete(y_test_Q3, deleteList)

predictor = trainer.train(np.array(X_train_Q7_50), np.array(y_train_Q7_50))

result = []
for row in X_test_Q7_50:
    rs=predictor.predict(row)
    result.append(rs)

get_accuracy(result,y_test_Q7_50)




print("Q7 reduce number of samples - 75%")

deleteList = random.sample(range(500), round(500 * 0.75))

X_train_Q7_75 = np.delete(X_train_Q3, deleteList, axis=0)
y_train_Q7_75 = np.delete(y_train_Q3, deleteList)

X_test_Q7_75 = np.delete(X_test_Q3, deleteList, axis=0)
y_test_Q7_75 = np.delete(y_test_Q3, deleteList)

predictor = trainer.train(np.array(X_train_Q7_75), np.array(y_train_Q7_75))

result = []
for row in X_test_Q7_75:
    rs=predictor.predict(row)
    result.append(rs)

get_accuracy(result,y_test_Q7_75)




print("Q7 reduce number of samples - 90%")

deleteList = random.sample(range(500), round(500 * 0.90))

X_train_Q7_90 = np.delete(X_train_Q3, deleteList, axis=0)
y_train_Q7_90 = np.delete(y_train_Q3, deleteList)

X_test_Q7_90 = np.delete(X_test_Q3, deleteList, axis=0)
y_test_Q7_90 = np.delete(y_test_Q3, deleteList)

predictor = trainer.train(np.array(X_train_Q7_90), np.array(y_train_Q7_90))

result = []
for row in X_test_Q7_90:
    rs=predictor.predict(row)
    result.append(rs)

get_accuracy(result,y_test_Q7_90)




print("Q7 reduce number of samples - 95%")

deleteList = random.sample(range(500), round(500 * 0.95))

X_train_Q7_95 = np.delete(X_train_Q3, deleteList, axis=0)
y_train_Q7_95 = np.delete(y_train_Q3, deleteList)

X_test_Q7_95 = np.delete(X_test_Q3, deleteList, axis=0)
y_test_Q7_95 = np.delete(y_test_Q3, deleteList)

predictor = trainer.train(np.array(X_train_Q7_95), np.array(y_train_Q7_95))

result = []
for row in X_test_Q7_95:
    rs=predictor.predict(row)
    result.append(rs)

get_accuracy(result,y_test_Q7_95)



print("Q8 Train the dual radial basis function machine to classify even vs. odd numbers")

