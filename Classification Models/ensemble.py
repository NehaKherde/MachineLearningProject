################################################################################################
# Ensemble model which pools predictions of CNN, MLP and non-linear SVM
# Prediction based on majority vote
# Note: This is a Python 3 code.
# Author: Maitrey Mehta
###############################################################################################

import numpy as np
import pandas as pd
from mlp import MLP
from non_linear_svm import nonLinearSVM
from cnn import CNN
from create_train_and_test import getTrainTest

test_data,train_data = getTrainTest("tagged_vectors_gen_dropshuff.npy","final_tagged_vectors_dropshuff.npy")
np.random.seed(15)
np.random.shuffle(train_data)
np.random.shuffle(test_data)
pred_mlp = MLP(train_data,test_data)
print(pred_mlp)
print(len(pred_mlp))
pred_svm = nonLinearSVM(train_data,test_data)
print(pred_svm)
print(len(pred_svm))
pred_cnn = CNN(train_data,test_data)
print(pred_cnn)
print(len(pred_cnn))
test_true_labels = test_data[:,-1]	
count=0
for i in range(0,len(test_true_labels)):
	if (pred_mlp[i]+pred_cnn[i]+pred_svm[i])>=2:
		pred =1
	else:
		pred=0 
	
	if pred ==test_true_labels[i]:
		count+=1
print("Test Acc: "+str(float(count)/len(test_true_labels)))