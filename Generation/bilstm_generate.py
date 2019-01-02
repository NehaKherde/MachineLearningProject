#############################################################################
# Module for generation BiLSTM to generate synthetic data
# Author: Maitrey Mehta
###############################################################################

import cPickle
from keras.models import load_model
import os
import numpy as np

print "Loading Vocabulary"
save_dir = "save"
vocab_file = os.path.join(save_dir,"words_vocab.pkl")

with open(os.path.join(save_dir,"words_vocab.pkl"),"rb") as f:
	words, vocab, vocabulary_inv = cPickle.load(f)

vocab_size = len(words)
print "Loading Model"
model = load_model(save_dir + "/" + 'my_model_gen_sentences.h5')

#To introduce some uncertainity, this function picks words that has lesser probability that the 
#best prediction word
#Temperature = 1,  probability for a word to be drawn is similar to the probability for the word 
#to be the next one in the sequence (the output of the word prediction model), 
#compared to other words in the dictionary
def sample(preds,temperature=1.0):
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds)/temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1,preds,1)
	return np.argmax(probas)

# This function ensures that the generated sequence
# is broken down into sentences.
def giveProperFakeSentences(generated):
	proper_generated=""
	first_fs_found = False
	for i in xrange(0,len(generated)):
		if generated[i] == ".":
			break
	prev_index=i+1
	for i2 in xrange(i+1,len(generated)):
		if generated[i2] =="." and (i2+1 - prev_index)<26:
			proper_generated =proper_generated + generated[prev_index:i2+1] + "\n"
			prev_index = i2 + 2
		elif generated[i2] =="." and (i2+1 - prev_index)>25:
			prev_index = i2 + 2
	return proper_generated





words_number = 1000
seed_sent_corp = np.load("sentences.npy")

for iteration in xrange(0,1000):
	seed_sentences = seed_sent_corp[iteration+2].lower()		#Seed sentence changed everytime for generation
	seq_legth = 24

	generated = ""
	sentence =[]

	for i in range(seq_legth):
		sentence.append("a")

	seed = seed_sentences.split()

	try:
		for i in range(len(seed)):
			sentence[seq_legth-i-1]=seed[len(seed)-i-1]
	except IndexError:
		continue
		
	generated += ' '.join(sentence)
	#Loop for generation
	for i in range(words_number):
		x=np.zeros((1,seq_legth,vocab_size))
		for t,word in enumerate(sentence):
			x[0,t,vocab[word]] = 1

		#calculate next word
		preds = model.predict(x,verbose=0)[0]
		next_index = sample(preds,0.33)
		next_word = vocabulary_inv[next_index]

		generated += " "+ next_word
		sentence = sentence[1:] + [next_word]
	proper_generated = giveProperFakeSentences(generated)
	print proper_generated
	text_file = open("Generated_BiLSTM.txt", "a")
	text_file.write(proper_generated)
	text_file.close()