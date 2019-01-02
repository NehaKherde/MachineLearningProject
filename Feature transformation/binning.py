#############################################################################################################
# Code to bin continuous feature values into discrete values                                                #
# Implemented binning by assigning bin value to a sorted feature column every time when the label changed.  #
# Author: Neha Kherde                                                                                       #
#############################################################################################################

from create_train_and_test import *
import pandas as pd

'''
Object to store bin values for each feature
'''
class Bin:
    def __init__(self, col_num):
        self.column_number = col_num
        self.min = 0
        self.max = 0
        self.bin_values = []


'''
Adds the transformed feature values into a new .csv file
'''
def add_data_to_csv(output_file_name, data):
    raw_data = {}
    col_array = []
    for col_num in range(0, 207):
        raw_data[col_num] = []
        col_array.append(col_num)

    # for col_index in range(len(data)-1):
    #     raw_data[col_index+1] = data[col_index]

    col_index = 1
    for col in data.T:
        print(col_index)
        if col_index != 207:
            raw_data[col_index] = col
            col_index += 1
        else:
            print("in 0")
            raw_data[0] = col
            col_index += 1

    #raw_data[0] = data[206]
    df = pd.DataFrame(raw_data, columns = col_array)
    df.to_csv(output_file_name)


def get_bin_class(data_cell_val, meta, index):
    bin_val_len = len(meta[index].bin_values)
    bin_val_copy = meta[index].bin_values[:]
    bin_val_copy.insert(0, 0)
    bin_val_copy.insert(len(bin_val_copy), 1)
    assign_val_arr = []
    for i in range(0, bin_val_len+1):
        assign_val_arr.append(i+1)

    for i in range(1, len(bin_val_copy)):
        if data_cell_val <= bin_val_copy[i]:
            return assign_val_arr[i-1]

def get_bin_class_divide_by_10(data_cell_val, meta, index):
    if meta[index].bin_values[9] ==  meta[index].bin_values[0]:
        return 1
    if data_cell_val >= meta[index].bin_values[9]:
        return 10
    if data_cell_val >= meta[index].bin_values[8]:
        return 9
    if data_cell_val >= meta[index].bin_values[7]:
        return 8
    if data_cell_val >= meta[index].bin_values[6]:
        return 7
    if data_cell_val >= meta[index].bin_values[5]:
        return 6
    if data_cell_val >= meta[index].bin_values[4]:
        return 5
    if data_cell_val >= meta[index].bin_values[3]:
        return 4
    if data_cell_val >= meta[index].bin_values[2]:
        return 3
    if data_cell_val >= meta[index].bin_values[1]:
        return 2
    if data_cell_val >= meta[index].bin_values[0]:
        return 1


def binning(data, meta):
    for data_row in data:
        for index in range(len(meta)):

            data_row[index] = get_bin_class(data_row[index], meta, index)
    return data


'''
params: complete corpus
Returns bin values for each feature
'''
def get_data_partitions(data):
    bin_data = []
    column_number = 0
    label_col = 206
    for column in data.T:

        data = data[data[:, column_number].argsort()]
        column_data = Bin(column_number)

        prev_row_feature = data[0][column_number]
        prev_row_label = data[0][label_col]
        zero_label_count = 0
        one_label_count = 0
        for index in range(1, len(data)): #index == row number
            # if feature values are same but have different labels, then maintain the count of unique labels
            if prev_row_feature == data[index][column_number]:
                if data[index][label_col] == 1:
                    one_label_count += 1
                else:
                    zero_label_count += 1
                prev_row_label = data[index][label_col]
            else:
                # if feature values are not same
                if zero_label_count != 0 or one_label_count != 0:
                    # from above if condition, assign the prev_row_label as the one with majority count
                    if zero_label_count > one_label_count:
                        prev_row_label = 0
                    else:
                        prev_row_label = 1
                    zero_label_count = 0
                    one_label_count = 0
                # If there is a change in label, then add a new bin value
                if prev_row_label != data[index][label_col]:
                    column_data.bin_values.append((data[index][column_number] + prev_row_feature)/2)
                prev_row_label = data[index][label_col]
            prev_row_feature = data[index][column_number]

        column_data.min = np.amin(column)
        column_data.max = np.amax(column)

        bin_data.append(column_data)
        column_number += 1

    # delete the last column as we don't need to split the label column.
    del bin_data[206]
    return bin_data


def get_data_partitions_divide_by_10(data):
    bin_data = []
    column_number = 0
    for column in data.T:
        column_data = Bin(column_number)
        column_data.min = np.amin(column)
        column_data.max = np.amax(column)

        # make bins with (min+max)/2
        increment_val = (column_data.min + column_data.max) / 10

        for i in range(10):
            val = column_data.min + (increment_val * i)
            column_data.bin_values.append(val)

        bin_data.append(column_data)
        column_number += 1

    # delete the last column as we don't need to split the label column.
    del bin_data[206]
    return bin_data


def main():

    incorrect_corpus_file = "final_tagged_vectors_srilm_wiki.npy"
    correct_corpus_file = "tagged_vectors_spacy.npy"

    test_data, train_data = get_test_train_data(correct_corpus_file, incorrect_corpus_file)
    train_data_partition = get_data_partitions(train_data)
    test_data_partition = get_data_partitions(test_data)

    train_data = binning(train_data, train_data_partition)
    test_data = binning(test_data, test_data_partition)

    add_data_to_csv("final_train_data_srilm_wiki_binning.csv", train_data)
    add_data_to_csv("final_test_data_srilm_wiki_binning.csv", test_data)



if __name__ == "__main__": main()