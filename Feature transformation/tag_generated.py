############################################################################################
# Module to assign preprocess and assign Part-of-Speech tags to generated sentences
# Author : Maitrey Mehta
############################################################################################

from tagger import tagger, vectorizeTagSeq
import numpy as np
import nltk

def convert2array(text):
	text = text.split("\n")
	textinwords=[]
	it=0
	for i in text:
		temp = i.split(" ")
		if temp[-1]==".":
			temp=temp[:len(temp)-1]
		temp = " ".join(temp)
		textinwords.append(temp)
	return np.array(textinwords)

if __name__=="__main__":
	text = open("incorrect_corpus.txt").read()
	x_words = convert2array(text)
	tagged_sequence = tagger(x_words)
	dimensions = np.load("dimensions.npy")
	dimension_array = np.load("dimension_array.npy")
	tagged_generated = vectorizeTagSeq(dimensions, tagged_sequence,dimension_array)
	np.save("tagged_vectors_gen_dropshuff.npy",np.array(tagged_generated))
	print len(tagged_sequence)
	print tagged_generated[1]
	#temp = np.load("tagged_vectors_gen_srilm_wiki.npy")	
	#print temp[1]