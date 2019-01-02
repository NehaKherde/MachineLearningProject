#####################################################################
# Code to split a dataset into test and train
# Author: Neha Kherde, Maitrey Mehta(getFinalTrainTest)
#####################################################################

import numpy as np
from random import shuffle

np.random.seed(15)
def write_dummy_data_into_npy_file(correct_sentences_file_name, fake_sentences_file_name):

    correct_data = np.load(correct_sentences_file_name)
    fake_data = np.load(fake_sentences_file_name)
    #print len(correct_data)
    #print len(fake_data)

    # find which of the 2 arrays / files have minimum number of records and consider only those many number of records.
    # Picking the records by shuffling
    # We choose 1208 as that was the minimal test size for a given generation method that was available to us
    np.random.shuffle(fake_data)
    fake_data = fake_data[:1208]

    np.random.shuffle(correct_data)
    correct_data = correct_data[:1208]

    # now separate few records from correct and fake data set into test_data
    count_of_test_data_from_each_file = int(len(correct_data) / 10)
    np.random.shuffle(correct_data)
    np.random.shuffle(fake_data)

    test_data1 = correct_data[:count_of_test_data_from_each_file]
    test_data2 = fake_data[:count_of_test_data_from_each_file]
    test_data = np.concatenate((test_data1, test_data2), axis=0)

    train_data1 = correct_data[count_of_test_data_from_each_file:]
    train_data2 = fake_data[count_of_test_data_from_each_file:]
    train_data = np.concatenate((train_data1, train_data2), axis=0)

    return test_data, train_data

def getFinalTrainTest(correct_sentences_file_name, fake_sentences_file_name,fake_file_test):

    correct_data = np.load(correct_sentences_file_name)
    fake_data = np.load(fake_sentences_file_name)
    fake_data_test = np.load(fake_file_test)
    # find which of the 2 arrays / files have minimum number of records and consider only those many number of records.
    # Picking the records by shuffling
    # We choose 1208 as that was the minimal test size for a given generation method that was available to us
    np.random.shuffle(fake_data)
    fake_data = fake_data[:1208]

    np.random.shuffle(correct_data)
    correct_data = correct_data[:2008]

    # now separate few records from correct and fake data set into test_data
    count_of_test_data_from_each_file = 800
    np.random.shuffle(correct_data)
    
    test_data1 = correct_data[:count_of_test_data_from_each_file]
    test_data2 = fake_data_test[:count_of_test_data_from_each_file]
    test_data = np.concatenate((test_data1, test_data2), axis=0)

    train_data1 = correct_data[count_of_test_data_from_each_file:]
    train_data2 = fake_data
    train_data = np.concatenate((train_data1, train_data2), axis=0)

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    return test_data, train_data



def getTrainTest(fake_file,fake_file_test=""):
    correct_corpus_file = "tagged_vectors_spacy.npy"			#This is the file containing vectors of real sentences
    fake_corpus_file = fake_file
    if fake_file_test == "":
        test_data, train_data = write_dummy_data_into_npy_file(correct_corpus_file , fake_corpus_file)
    else:
        test_data, train_data = getFinalTrainTest(correct_corpus_file , fake_corpus_file, fake_file_test)
    return test_data,train_data

if __name__ == "__main__": 
    test_data,train_data=getTrainTest("tagged_vectors_gen_dropshuff.npy","final_tagged_vectors_dropshuff.npy")
    #print len(train_data)
    #print len(test_data)