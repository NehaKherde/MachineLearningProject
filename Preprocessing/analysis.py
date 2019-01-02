######################################################################################
# Code for analysis of training set. 
# Objectives -
# 1) Stats on most frequently occuring words
# 2) The maximum length of a sentence in the training set
# 3) The maximum occurence of a pos tag in a sentence
# 4) Size of corpus
#
# Author- Maitrey Mehta
######################################################################################

#Function gives number of words and sentences in the corpus before preprocessing
def sizeOfCorpus(filename):
	wordcount=0
	x = open(filename,"r").read()
	x_words = x.split(" ")
	print len(x_words)					# Total words: 2088629
	x_lines = x.split(". ")
	print len(x_lines)					# Total sentences: 83398    Average words in a sentence: 25.044

	return x_words, x_lines

def wordDistribution(words):
	dist={}
	for i in words:
		print i
		if i not in dist.keys():
			dist[i]=0
		else:
			dist[i]+=1
	d_view = [ (v,k) for k,v in dist.iteritems() ]
	d_view.sort(reverse=False)										 # natively sort tuples by first element
	count=0
	for v,k in d_view:
		print "%s: %d" % (k,v)										# <unk>: 54624 (5th after , . the of) @-@:16905
		count+=1
	print count														#Vocab Size: 33279


if __name__=="__main__":
	filename="./../Data/wiki.train.tokens"
	words, sentences = sizeOfCorpus(filename)
	wordDistribution(words)