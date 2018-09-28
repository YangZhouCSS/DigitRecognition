#digit recognition using Naive Bayes
#Yang Zhou

import random
random.seed(12345)

#read traning file
lines_to_use = 3000   #how many lines of trainset to use for each class. max = 5000
trainingSet = []

for i in range(10):
    file = "train" + str(i) + ".txt"
    data = open(file).readlines()
    data = random.sample(data, lines_to_use) 
    trainingSet = trainingSet + data
    
for i in range(len(trainingSet)):
    trainingSet[i] = [float(x) for x in trainingSet[i].split(" ")]
    
#read testing file
testSet = []

for i in range(10):
    file = "test" + str(i) + ".txt"
    data = open(file).readlines()
    data = random.sample(data, 200)   #test set uses 200 from each digit
    testSet = testSet + data
    
for i in range(len(testSet)):
    testSet[i] = [float(x) for x in testSet[i].split(" ")]




#Replace all nonzero entries with 1 
for i,line in enumerate (trainingSet):
    the_class=trainingSet[i][0]
    for j,x in enumerate(line):
        if x > 0:
            line[j] = 1
        else:
            line[j] = 0
    trainingSet[i] = line
    trainingSet[i][0] = the_class

for i,line in enumerate (testSet):
    the_class=testSet[i][0]
    for j,x in enumerate(line):
        if x > 0:
            line[j] = 1
        else:
            line[j] = 0
    testSet[i] = line
    testSet[i][0] = the_class



print ("number of traning data/ test data:")
print(len(trainingSet), len(testSet))




#create a matirx of probabilities


prob = {}
t1 = 0
t2 = lines_to_use - 1
for class_number in range(10):
    trainingSet_transposed = list(map(list, zip(*trainingSet[t1:t2])))
    del trainingSet_transposed[0]
    prob[class_number] = []
    for attribute in trainingSet_transposed:
        prob[class_number].append( attribute.count(1) / float(lines_to_use))  #probability of getting 1 in this cell given class_number
    t1 += lines_to_use
    t2 += lines_to_use
        
    


#classify test data
results = []

for i,line in enumerate(testSet):
    line = line[1:]
    class_p = []
    for class_number in range(10):
        p_this_class = 1
        for i,attribute in enumerate(line):
            if attribute == 1:
                p = prob[class_number][i]
            else:
                p = 1 - prob[class_number][i]

            p_this_class = p * p_this_class
            
            
            
        class_p.append(p_this_class)
            

            
    results.append(class_p.index(max(class_p)))
        
            
count = 0            
for i,line in enumerate(testSet):
    if int(line[0]) == results[i]:
        count += 1

accuracy = count / len(testSet)


print ("overall accuracy:" + str(accuracy))


#confusion matrix
from sklearn.metrics import confusion_matrix

y_true = []
for i,line in enumerate(testSet):
    y_true.append(line[0])

y_pred = results

cm=confusion_matrix(y_true, y_pred)

print("confusion matrix:")
print(cm)
##
##accuracy_each =[]
##for i,line in enumerate(cm):
##    score = line[i] / sum(line)
##    score = int(score * 10000) / 10000.0
##    accuracy_each.append(score)
##print(accuracy_each)
