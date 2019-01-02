######################################################################################
# Code for preprcoessing of training set. 
# Objectives -
# 1) Removing unwanted tokens
# 2) Sentence separation
# Author- Maitrey Mehta
######################################################################################
import numpy as np

def removeWords(filename):
	wordcount=0
	x = open(filename,"r").read()
	x_lines = x.split(". ")
	x_words=[]
	for i in x_lines:
		temp = i.split(" ")
		x_words.append(temp)
		
	noofwords=0
	x_words = np.array(x_words)
	emptysent=[]
	for i in xrange(0,len(x_words)):
		ind2remove=[]
		for j in xrange(0,len(x_words[i])):
			if (x_words[i][j]=='.') or (x_words[i][j]=='<unk>') or (x_words[i][j]=='@-@') or (x_words[i][j]==',') or (x_words[i][j]=='') or (x_words[i][j]=='\n'):
				ind2remove.append(j)
		x_words[i]=np.delete(x_words[i],ind2remove)
		noofwords+=len(x_words[i])
		if(len(x_words[i])==0):
			emptysent.append(i)

	x_words = np.delete(x_words,emptysent)						# New number of sentences: 77749
	print x_words												#New number of words: 1807067
	np.save("words.npy",x_words)
	return x_words


if __name__=="__main__":
	filename="./../Data/wiki.train.tokens"
	words = removeWords(filename)
