###################################################################################################################
# Code to implement Non-linear SVM using scikit-learn's SVM(libsvm)  
# Note that this module is implemented in python3 since there are bugs in the python2 sklearn svm package
# Author: Maitrey Mehta
###################################################################################################################

from sklearn import svm
from create_train_and_test import getTrainTest
import numpy as np
import pandas as pd

def column(matrix, i):
    return [row[i] for row in matrix]

def nonLinearSVM(train_data,test_data):
	# Training block
	feature_train = train_data[:,:-1]
	true_labels = train_data[:,-1]	

	array=[]
	for ep in range(800,801):
		clf =  svm.SVC(gamma='scale',C=100,kernel="rbf",max_iter=ep,random_state=1)
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
		temp.append(ep)
		temp.append(acc2*100)
		temp.append(acc*100)
		print(temp)
		array.append(temp)
	return predicted

if __name__=="__main__":
	test_data,train_data = getTrainTest("tagged_vectors_gen_dropshuff.npy","final_tagged_vectors_dropshuff.npy")
	np.random.seed(15)
	np.random.shuffle(train_data)
	np.random.shuffle(test_data)
	pred = nonLinearSVM(train_data,test_data)
	

# array = np.array(array)
# df = pd.DataFrame({"Epoch":column(array,0),"Train":column(array,1),"Test":column(array,2)})
# df.to_csv("Acc_non-linear_svm.csv")