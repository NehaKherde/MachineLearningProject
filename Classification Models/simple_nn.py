########################################################
# Impelentation of 2-layer Neural Network from scratch
# Author: Maitrey Mehta
#########################################################

import numpy as np
from create_train_and_test import getTrainTest
from sklearn.metrics import classification_report, confusuion_matrix

# Function that return the sigmoid of the value
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def column(matrix,i):
	return [row[i] for row in matrix]

def crossEntropyLoss(y,y_dash):
	length = y.shape[0]
	loss = (-1/length)*(np.sum(np.multiply(np.log(y_dash),y))+np.sum(np.multiply(np.log(1-y_dash),(1-y))))
	return loss	

test_data,train_data = getTrainTest("tagged_vectors_gen_dropshuff.npy")
np.random.seed(15)
np.random.shuffle(train_data)
np.random.shuffle(test_data)

feature_train = train_data[:1500,:-1]
true_labels = train_data[:1500,-1]
# We need to reshape the vector into a matrix form
true_labels = np.reshape(true_labels,(len(true_labels),1))

no_features = feature_train.shape[1]
hidden_units = 5
eta = 1

w1 = np.random.randn(hidden_units,no_features)
b1 = np.zeros((hidden_units,1))
w2 = np.random.randn(1,hidden_units)
b2 = np.zeros((1,1))
examples = feature_train.shape[0]

array=[]
epochs = 5000
for i in range(5000):
	# Feed-forward
	o1 = np.matmul(w1,feature_train.T) + b1
	act1 = sigmoid(o1)
	o2 = np.matmul(w2,act1) + b2
	act2 = sigmoid(o2)

	# Loss calcualtion
	loss = crossEntropyLoss(true_labels,act2.T)
	
	# Backpropagation	
	diffo2 = act2.T - true_labels
	diffw2 = (1./m)*np.matmul(diffo2.T,act1.T)
	diffb2 = (1./m)*np.sum(diffo2.T,axis=1,keepdims=True)
	diffact1 = np.matmul(w2.T,diffo2.T)
	diffo1 = diffact1*sigmoid(o2)*(1-sigmoid(o2))
	diffw1 = (1./m)*np.matmul(diffo1.T,feature_train)
	diffb1 = (1./m)*np.sum(diffo1.T,axis=1,keepdims=True)

	# Weight updates
	w2 -= eta*diffw2
	w1 -= eta*diffw1	
	b1 -= eta*diffb1
	b2 -= eta*diffb2

	# Testing on Train data
	feature_test = train_data[:,:-1]
    test_true_labels = train_data[:,-1]	
    test_true_labels = np.reshape(test_true_labels,(len(test_true_labels),1))
    o1 = np.matmul(w1, feature_test.T) + b1
    act1 = sigmoid(o1)
    o2 = np.matmul(w2, act1) + b2
    act2 = sigmoid(o2)

    predictions = (act2>.5)[0,:]
    labels = (test_true_labels.T == 1)[0,:]
    acc2 = (confusion_matrix(predictions, labels)[0][0] +confusion_matrix(predictions, labels)[1][1])/float(len(test_true_labels))

    # Testing on Train data
    feature_test = test_data[:,:-1]
    test_true_labels = test_data[:,-1]	
    test_true_labels = np.reshape(test_true_labels,(len(test_true_labels),1))
    o1 = np.matmul(w1, feature_test.T) + b1
    act1 = sigmoid(o1)
    o2 = np.matmul(w2, act1) + b2
    act2 = sigmoid(o2)

    predictions = (act2>.5)[0,:]
    labels = (test_true_labels.T == 1)[0,:]

    # Printing confusion matrix for analysis
    print confusion_matrix(predictions, labels)
    acc = (confusion_matrix(predictions, labels)[0][0] +confusion_matrix(predictions, labels)[1][1])/240.0
    #print classification_report(predictions, labels)
    if (i+1)%10 == 0:
		temp=[]
		temp.append(i+1)
		temp.append(acc2*100)
		temp.append(acc*100)
		print temp
		array.append(temp)

df = pd.DataFrame({"Epoch":column(array,0),"Train":column(array,1),"Test":column(array,2)})
df.to_csv("Acc.csv")