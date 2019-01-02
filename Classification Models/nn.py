###################################################################################################################
# A class based Neural Network aproach 
# NOTE: This was not finally used since it doesn't converge 
# Author: Maitrey Mehta
###################################################################################################################
import numpy as np
from create_train_and_test import getTrainTest
import keras
import math
import tensorflow as tf

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))
def sigmoid_derivative(x):
	return x*(1.0-x)

def compute_loss(Y, Y_hat):
    m = Y.shape[1]
    L = -(1./m) * ( np.sum( np.multiply(np.log(Y_hat),Y) ) + np.sum( np.multiply(np.log(1-Y_hat),(1-Y)) ) )
    return L

class NN:
	def __init__(self,data,label):
		self.input = data
		self.weights1 = np.random.rand(self.input.shape[1],5)
		self.weights2 = np.random.rand(5,1)
		self.y = label
		self.output = np.zeros(self.y.shape)

	def feedforward(self):
		self.layer1 = sigmoid(np.dot(self.input,self.weights1))
		self.layer2 = sigmoid(np.dot(self.layer1,self.weights2))
		return self.layer2

	def backprop(self):
		#print (self.y/(self.output+np.exp(-20))+(1-self.y)/(1-self.output+np.exp(-20)))*sigmoid_derivative(self.output)
		d_weights2 = np.dot(self.layer1.T, (-(self.y)/(self.output-np.exp(-20))+(1-self.y)/(1-self.output+np.exp(-20)))*sigmoid_derivative(self.output))
		d_weights1 = np.dot(self.input.T, np.dot(-(self.y/(self.output+np.exp(-20))+(1-self.y)/(1-self.output+np.exp(-20)))*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
    
		self.weights1 += d_weights1
		self.weights2 += d_weights2
	

	def train(self,data,label):
		self.output = self.feedforward()
		print self.output
		self.backprop()

def test(test_features,test_true_label,network):
	network2 = NN(test_features,test_true_label)
	network2.weights1 = network.weights1
	network2.weights2 = network.weights2
	error = np.abs(test_true_label - network2.feedforward())
	print network2.feedforward()
	print np.average(error)


if __name__=="__main__":
	test_data,train_data = getTrainTest("tagged_vectors_gen_dropshuff.npy")
	np.random.seed(15)
	np.random.shuffle(train_data)
	np.random.shuffle(test_data)
	
	feature_train = train_data[:,:-1]
	true_labels = train_data[:,-1]	
	true_labels = np.reshape(true_labels,(len(true_labels),1))
	X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
	y=np.array(([0],[1],[1],[0]), dtype=float)
	#network = NN(feature_train,true_labels)
	network = NN(X,y)
	epochs = 500
	for i in xrange(0,epochs):
		if i%2 == 0:
			print "Epoch " + str(i+1)
			#print "Loss: " + str(np.mean(-true_labels*np.log(network.feedforward()+np.exp(-12)) - (1-true_labels)*np.log(network.feedforward()+np.exp(-12))))
			print "Loss: " + str(np.mean(-y*np.log(network.feedforward()+np.exp(-12)) - (1-y)*np.log(network.feedforward()+np.exp(-12))))

		#network.train(feature_train,true_labels)
		network.train(X,y)
	# np.save("weights_l1.npy",network.weights1)
	# np.save("weights_l2.npy",network.weights1)

	# feature_test = test_data[:,:-1]
	# test_true_labels = test_data[:,-1]	
	# test_true_labels = np.reshape(test_true_labels,(len(test_true_labels),1))
	# test(feature_test,test_true_labels,network)