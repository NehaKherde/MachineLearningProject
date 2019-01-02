#####################################################################
# Code to create fake/corrupt sentences by dropping and shuffling words
# Author: Neha Kherde
#####################################################################

import numpy as np
import random
import re


def read_sentences():

    file_contents = np.load('words.npy')
    sentence_list = []
    for sentence in file_contents:
        sentence = sentence.tolist()
        if len(sentence) > 2:
            sentence_list.append(sentence)
    return sentence_list


def create_fake_sentence_using_swap(sentence):
    sentence_length = len(sentence)
    dummy_sentence = list(sentence)
    for times in range(3):
        random_numbers = random.sample(range(0, sentence_length), 2)
        ignored_characters = "?.!&-"
        if len(sentence) > 2:
            while (sentence[random_numbers[0]] != '' and sentence[random_numbers[0]] in ignored_characters) or \
                    (sentence[random_numbers[1]] != '' and sentence[random_numbers[1]] in ignored_characters):
                random_numbers = random.sample(range(0, sentence_length), 2)
            dummy_sentence[random_numbers[0]], dummy_sentence[random_numbers[1]] = dummy_sentence[random_numbers[1]], dummy_sentence[random_numbers[0]]
    return dummy_sentence


def create_fake_sentence_using_drop(sentence):
    #sentence_length = len(sentence)
    dummy_sentence = list(sentence)
    print(sentence)
    for times in range(3):
        print(times)
        random_number = random.randint(0, len(dummy_sentence)-1)
        ignored_characters = "?.!&-"
        while dummy_sentence[random_number] in ignored_characters and len(dummy_sentence) > 3:
            random_number = random.randint(0, len(dummy_sentence) - 1)
        del dummy_sentence[random_number]
    return dummy_sentence


def add_fake_sentence_to_list(fake_sentence, fake_sentence_list):
    sentence = ""
    for word in fake_sentence:
        sentence += word
        sentence += ' '
    fake_sentence_list.append(sentence)


def generate_fake_sentences(sentence_list):
    fake_sentence_list = []
    for sentence in sentence_list:
        if len(sentence) > 1:
            fake_sentence = create_fake_sentence_using_swap(sentence)
            add_fake_sentence_to_list(fake_sentence, fake_sentence_list)

            if len(sentence) > 3:
                fake_sentence = create_fake_sentence_using_drop(sentence)
                add_fake_sentence_to_list(fake_sentence, fake_sentence_list)

    return fake_sentence_list


def write_fake_sentences_to_file(fake_sentence_list):
    with open('incorrect_corpus.txt', 'w') as f:
        for sentence in fake_sentence_list:
            f.write("%s\n" % sentence)


def main():
    sentence_list = read_sentences()
    fake_sentence_list = generate_fake_sentences(sentence_list)
    write_fake_sentences_to_file(fake_sentence_list)


if __name__ == "__main__": main()
