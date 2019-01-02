#############################################################################
# Module for training BiLSTM to generate synthetic data
# Author: Maitrey Mehta
###############################################################################


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, LSTM, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy

import numpy as np 
import random
import sys
import os
import time
import codecs
import collections
import pickle as cPickle

data_dir = "Output.txt"			# Source File of training data
save_dir = "save"				# Directory where the models are saved
vocab_file = os.path.join(save_dir,"words_vocab.pkl")

# Function that converts words in the corpus to a list of words
def create_wordlist(doc):
	w1=[]
	data=doc.split(" ")
	for word in data:
		w1.append(word.lower())
	return w1

data = open("Output.txt","r").read()
data=data[:len(data)/50]				#We choose to cut short the file to 1/50th size due to memory constraints
wordlist = create_wordlist(data)		

word_counts = collections.Counter(wordlist)			

#Mapping from index to word
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))


#Mapping word to index
vocab = {x: i for i,x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]

vocab_size = len(words)
with open(os.path.join(vocab_file),"wb") as f:
	cPickle.dump((words,vocab,vocabulary_inv),f)

sequences=[] 	#Sequence of words
next_words = []	#The list contains the next words for each sequences of the sequences list
seq_length=24	#Consider 25 length sequences
sequences_step=1 #Next sequence is considered after skipping sequence_step -1 words

# This creates the list of training sequences and the true next word
for i in range(0,len(wordlist)-seq_length,sequences_step):
	sequences.append(wordlist[i:i+seq_length])
	next_words.append(wordlist[i+seq_length])

# This section just converts the word list to a list of integers
X = np.zeros((len(sequences),seq_length,vocab_size),dtype = np.bool)
y = np.zeros((len(sequences),vocab_size),dtype=np.bool)
for i,sentence in enumerate(sequences):
	for t,word in enumerate(sentence):
		X[i,t,vocab[word]] = 1
	y[i,vocab[next_words[i]]] = 1

#Function that creates the architecture of Bi-LSTM
def bidirectional_lstm_model(seq_length,vocab_size):
	print("Building LSTM Model")
	model = Sequential()
	model.add(Bidirectional(LSTM(rnn_size,activation="relu"),input_shape=(seq_length,vocab_size))) 
	model.add(Dropout(0.6))
	model.add(Dense(vocab_size))
	model.add(Activation("softmax"))

	optimizer = Adam(lr=learning_rate)
	callbacks = [EarlyStopping(patience=2,monitor="val_loss")]
	model.compile(loss="categorical_crossentropy",optimizer=optimizer, metrics = [categorical_accuracy])
	print("Model Built")
	return model

rnn_size = 256
learning_rate = 0.001

md = bidirectional_lstm_model(seq_length,vocab_size)

batch_size=32
num_epochs=50

callbacks = [EarlyStopping(patience=4,monitor="val_loss"),
			ModelCheckpoint(filepath=save_dir+ "/" + 'my_model_gen_sentences.{epoch:02d}-{val_loss:.2f}.hdf5',monitor="val_loss",verbose=0,mode="auto",period=2)]

history = md.fit(X,y,batch_size=batch_size,shuffle=True,epochs=num_epochs,callbacks=callbacks,validation_split=0.1)

md.save(save_dir+"/"+'my_model_gen_sentences.h5') #We save this final model