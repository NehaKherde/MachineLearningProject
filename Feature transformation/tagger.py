######################################################################################
# Code for obtaining vectorized POS tagged sequences of training set. 
# This is the module which does feature transformation
# Author- Maitrey Mehta
######################################################################################

import numpy as np
import nltk
import spacy

#Tags the sentences. Takes sentences less than 40 words
def tagger(x_words):
	nlp = spacy.load('en_core_web_sm')
	tagged_sequence = []
	max_length=0
	count_more_than_25=0
	for i in x_words:
		print i
		#tagged = nltk.pos_tag(i)
		try:
			tagged = nlp(unicode(i))
		except UnicodeDecodeError:
			continue
		#print unicode(i)
		tagged_sent = [j.pos_ for j in tagged if j.pos_ != u'PUNCT' and j.pos_ != u'SPACE']		#ignores spaces and punctuations
		print tagged_sent
		leng = len(tagged_sent)
		if leng>40:
			count_more_than_25+=1
		if leng>max_length:
			max_length = leng	
		if leng<=40 and leng!=0:
			tagged_sequence.append(tagged_sent)

	print "Maximum Tagged Length: " + str(max_length)	#Max Length: 564			#Noofsentences less than 25 words: 49620
	print "Sentences More than 40: "+ str(count_more_than_25)
	return tagged_sequence

######This block is based on old features
# def taggedAnalysis(tagged_sequence):
# 	tagsdict={}
# 	for i in tagged_sequence:
# 		temp_dict={}
# 		for j in i:
# 			if j not in temp_dict.keys():
# 				temp_dict[j]=1
# 			else:
# 				temp_dict[j]+=1

# 		for key,value in temp_dict.iteritems():
# 			if key not in tagsdict.keys():
# 				tagsdict[key]=value
# 			elif value>tagsdict[key]:
# 				tagsdict[key]=value
# 	dimensions=0
# 	dimension_array=[]
# 	for key,value in tagsdict.iteritems():
# 		dimensions+=value
# 		for l in xrange(1,value+1):
# 			dimension_array.append(str(key)+str(l))
# 	return dimensions, dimension_array, tagsdict


# This is an analysis block which determines all the dimensions possible based on training data
# The dimensions are determined based on training set and an OTHER dimension is added to accomodate
# values that cannot be assigned to any dimension
def taggedAnalysis(tagged_sequence):
	dimension_array=[]
	dimensions = 0
	for i in tagged_sequence:
		for j in xrange(0,len(i)):
			if j==0:
				dim = "$|"+i[j]
			else:
				dim = i[j-1]+"|"+i[j]
			if dim not in dimension_array:
					dimension_array.append(dim)
	dimension_array.append("OTHER")
	dimensions = len(dimension_array)
	
	return dimensions,dimension_array


# This block converts the list of words(sentences) into sentence vectors
# based on the dimensions decided in the previous function
def vectorizeTagSeq(dimensions,tagged_sequence, dimension_array,label=False):
	tagged_vectors = []
	for i in tagged_sequence:
		temp_dict={}
		array= np.zeros(dimensions)
		for j in xrange(0,len(i)):
			if j==0:
				key = "$|"+i[j]
			else:
				key = i[j-1]+"|"+i[j]
			if key not in temp_dict.keys():
				if key not in dimension_array:
					if "OTHER" not in temp_dict.keys():
						temp_dict["OTHER"]=1
					else:
						temp_dict["OTHER"]+=1
				else:
					temp_dict[key]=1
			else:
				temp_dict[key]+=1

			# if j not in temp_dict.keys():
			# 	temp_dict[j]=1
			# 	array.append(j+str("1"))
			# else:
			# 	temp_dict[j]+=1
			# 	array.append(j+str(temp_dict[j]))

		for j in temp_dict.keys():
			index = np.where(dimension_array==j)
			index = index[0][0]
			array[index] = temp_dict[j]
	

		array = np.true_divide(array,len(i))
		if label ==True:
			array = np.append(array,[1])
		else:
			array = np.append(array,[0])

		# for tag1 in xrange(0,len(array)):
		# 	if array[tag1] in dimension_array:
		# 		index = dimension_array.index(array[tag1])
		# 		temp_vec[index] = tag1+1
		
		

			
		tagged_vectors.append(array)
	return tagged_vectors



if __name__ == "__main__":
	#x_words = np.load("words.npy")
	#tagged_sequence = tagger(x_words)
	#print tagged_sequence
	#np.save("tagged_spacy.npy",np.array(tagged_sequence))
	tagged_sequence = np.load("tagged_spacy.npy")
	# dimensions, dimension_array = taggedAnalysis(tagged_sequence)
	# print dimensions				#206
	# print dimension_array
	# np.save("dimensions.npy",np.array(dimensions))
	# np.save("dimension_array.npy",np.array(dimension_array))
	
	dimensions = np.load("dimensions.npy")
	dimension_array = np.load("dimension_array.npy")
	tagged_vectors = vectorizeTagSeq(dimensions,tagged_sequence,dimension_array,True)
	#print tagged_vectors
	np.save("tagged_vectors_spacy.npy",np.array(tagged_vectors))
	#tagged_vectors = np.load("tagged_vectors.npy")
	# words = np.load("words.npy")
	# str_words = ""
	# sentence=[]
	# for innerlist in words:
	# 	sent=""
	# 	for item in innerlist:
	# 		if sent!="":
	# 			sent = sent + " "
	# 		sent = sent + str(item)
	# 	sentence.append(sent + " . ")	 
	# 	str_words = str_words + sent + " . "
	# print sentence 
	# np.save("sentences.npy",np.array(sentence))
	# # text_file = open("Output.txt", "w")
	# text_file.write(str_words)
	# text_file.close()


############SPACY TAGS ##############################
# ADJ: adjective
# ADP: adposition
# ADV: adverb
# AUX: auxiliary verb
# CONJ: coordinating conjunction
# DET: determiner
# INTJ: interjection
# NOUN: noun
# NUM: numeral
# PART: particle
# PRON: pronoun
# PROPN: proper noun
# PUNCT: punctuation
# SCONJ: subordinating conjunction
# SYM: symbol
# VERB: verb
# X: other
# 17 tags = 289 combinations
#####################################################