############################################################################################################
# Implementation of SVM  
# Author: Maitrey Mehta
#############################################################################################################

import numpy as np
import math
from create_train_and_test import getTrainTest
seed=5
#Function that return prediction based on the data instance sent and the current value of weights
def predictVal(instance, weights):
	#Calculating sum(wi*xi)+b
	prediction = weights[-1]
	for i in range(0,len(instance)-1):
		prediction += weights[i]*instance[i]
	#Emulating Sign Function
	if instance[-1]==1:
		return prediction
	else:
		return -prediction


#Function that trains SVM weights
def trainSVM(data,eta,epochs,C):
	eta_zero =eta
	net_error=0
	np.random.seed(seed)
	init_weights = np.zeros(len(data[0]))
	weights = init_weights
	#print(weights)
	array=[]
	for ep in range(0,epochs):
		net_error=0
		np.random.shuffle(data)
		eta = eta_zero/(1+ep)
		for i in range(0,len(data)):
			#print(weights[0])
			if data[i][-1]==0:
				y = -1
			else:
				y=1

			prediction = predictVal(data[i],weights)	#Getting prediction
			if prediction <= 1:
				net_error += 1										#Incrementing mistake
				for j in range(0,len(weights)-1):
					weights[j] = (1-eta)*weights[j]+ eta*C*y*data[i][j]		#Update weights
				weights[-1] = (1-eta)*weights[-1] + eta*C*y							#Update Bias
			else:
				for j in range(0,len(weights)-1):
					weights[j] = (1-eta)*weights[j]		#Update weights
				weights[-1] = (1-eta)*weights[-1]
	 
	return weights


def testSVM(data,weights):
	error = 0
	cmatrix = np.zeros((2,2))
	
	for i in range(0,len(data)):
		prediction = weights[-1]	#Getting prediction
		for k in range(0,len(data[i])-1):
			prediction += weights[k]*data[i][k]
		#print(prediction)
		if prediction >= 0 and data[i][-1]==1:
			cmatrix[0][0]+=1
		elif prediction >= 0 and data[i][-1]==0:
			cmatrix[1][0]+=1
		elif prediction < 0 and data[i][-1]==0:
			cmatrix[1][1]+=1
		else:
			cmatrix[0][1]+=1
	#print(cmatrix)
	accuracy = float((cmatrix[0][0]+cmatrix[1][1]))/len(data)
		
	return accuracy



#Function for tasks of the SVM Task
def SVM(data_train,data_test,dev_file):
	print("***********Section for training SVM*************** ")
	global seed
	eta_vals = [1,0.1,0.01,0.001]
	Cs = [1000,100,10,1,0.1,0.01,0.001]
	
	accs=np.zeros((len(eta_vals),len(Cs)),dtype=float)
	
	develop_test,develop_train = getTrainTest(dev_file)
	print len(develop_test)
	epochs = 1000
	#Getting the best learning rate by testing on developset
	for i in range(0,len(eta_vals)):
		for j in range(0,len(Cs)):
			print("Development Test for eta value: " + str(eta_vals[i]) + "\t and C: "+str(Cs[j]))
			weights_dev = trainSVM(develop_train,eta_vals[i],epochs,Cs[j])
			accs[i][j] = testSVM(develop_test,weights_dev)	
			print("Development Accuracy: " + str(accs[i][j]))

	indexs  = np.argwhere(accs.max() == accs)[0]		#Gets the index of the max value
	best_eta = eta_vals[indexs[0]]
	best_C = Cs[indexs[1]]
	print accs
	print("Best Crossvalidation Accuracy is of values: ")
	print("Eta: " + str(best_eta) + " C: "+ str(best_C) +" with accuracy of: " + str(accs[indexs[0]][indexs[1]]) + "\n")

	# best_eta = 0.1
	# best_C = 1000
	eta=best_eta
	weights = trainSVM(data_train,eta,epochs,best_C)
	
	accuracy = testSVM(data_test,weights)
	#Testing on diabetes.test based on the best weights
	print("\nTest Accuracy for SVM Model on Test set: " + str(accuracy))
	print("******************************************************\n\n ")


if __name__=="__main__":
	np.random.seed(42)
	#print weights	
	test_data,train_data = getTrainTest("tagged_vectors_gen_dropshuff.npy","final_tagged_vectors_dropshuff.npy")
	SVM(train_data,test_data,"tagged_vectors_gen_srilm_wiki.npy")