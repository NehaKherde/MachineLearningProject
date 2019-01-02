########################################################################
# Implementation of Convolutional Neural Network using Keras
# Author : Maitrey Mehta	
#####################################################################

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from create_train_and_test import getTrainTest

def CNN(train_data,test_data):
	batch_size = 128
	num_classes = 2
	epochs = 110

	x_train = train_data[:,:-1]															#The last value is the true label
	x_test = test_data[:,:-1]
	y_train = train_data[:,-1]
	y_test = test_data[:,-1]

	# Since we had 206 feature, it only has 1,2 and 103 as its factors.
	# A 103x2 vector was not enough to effeciently run the convolutional
	# window and hence we pad two more zeros and reshape it into a 26x8 
	# vector later
	zeros_to_append_train = np.zeros((len(x_train),2))					
	zeros_to_append_test = np.zeros((len(x_test),2))
	x_train = np.append(x_train,zeros_to_append_train,axis=1)
	x_test = np.append(x_test,zeros_to_append_test,axis=1)

	x_train = x_train.reshape(len(x_train),26,8,1)
	x_test = x_test.reshape(len(x_test),26,8,1)

	# print('x_train shape:', x_train.shape)
	# print(x_train.shape[0], 'train samples')
	# print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	# Using keras functions to build the architextire
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=(26,8,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.75))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])

	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(x_test, y_test))
	predictions = model.predict_classes(x_test)
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	return predictions

if __name__=="__main__":
	# the data, split between train and test sets
	test_data, train_data = getTrainTest("tagged_vectors_gen_bilstm.npy","final_tagged_vectors_bilstm.npy")			#Note this function is in create_train_and_test
	pred = CNN(train_data,test_data)