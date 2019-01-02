###################################################################################################################
# Code to test scikit-learn's Multi-Layer Perceptron algorithm. 
# This was done as a comapritive measure to the one we implemented from scatch.
# Author: Maitrey Mehta
###################################################################################################################

from sklearn.neural_network import MLPClassifier
from create_train_and_test import getTrainTest
import numpy as np
import pandas as pd

def column(matrix, i):
    return [row[i] for row in matrix]

def MLP(train_data,test_data):
	# Training block
	feature_train = train_data[:,:-1]
	true_labels = train_data[:,-1]	
	array=[]
	for i in range(35,36):
		clf =  MLPClassifier(solver='lbfgs', alpha=1, hidden_layer_sizes=(5), random_state=1,max_iter=i,activation="tanh",learning_rate='invscaling',learning_rate_init=1)
		clf.fit(feature_train,true_labels)
		
		# Testing block
		# This block gets the training accuracy
		feature_test = train_data[:,:-1]
		test_true_labels = train_data[:,-1]	
		predicted = clf.predict(feature_test)
		count=0
		for i in range(0,len(test_true_labels)):
			if predicted[i]==test_true_labels[i]:
				count+=1
		print("Train Acc: "+str(float(count)/len(test_true_labels)))
		acc2 = float(count)/len(test_true_labels)

		# This block gets the testing accuracy
		feature_test = test_data[:,:-1]
		test_true_labels = test_data[:,-1]	
		predicted = clf.predict(feature_test)
		count=0
		for i in range(0,len(test_true_labels)):
			if predicted[i]==test_true_labels[i]:
				count+=1
		print("Test Acc: "+str(float(count)/len(test_true_labels)))
		acc = float(count)/len(test_true_labels)

		temp=[]
		temp.append(i)
		temp.append(acc2*100)
		temp.append(acc*100)
		print(temp)
		array.append(temp)
	return predicted


if __name__=="__main__":
	test_data,train_data = getTrainTest("tagged_vectors_gen_srilm_wiki.npy","final_tagged_vectors_srilm_wiki.npy")
	np.random.seed(15)
	np.random.shuffle(train_data)
	np.random.shuffle(test_data)
	pred = MLP(train_data,test_data)
	
	

# array = np.array(array)
# df = pd.DataFrame({"Epoch":column(array,0),"Train":column(array,1),"Test":column(array,2)})
# df.to_csv("Acc_mlp.csv")