#####################################################################
# Implementation of Bagging using Decision Trees
# Cross validation using multiple classifiers and by limiting height
# Author: Neha Kherde
#####################################################################

from data import Data
import numpy as np
import random
import math
DATA_DIR = 'data_files/'
training_file = "train_data_srilm_wiki_binning.csv"
height = 15
#data = np.loadtxt(DATA_DIR + 'train_data.csv', delimiter=',', dtype = str)
#data_obj = Data(data = data)


test_obj = Data(fpath = DATA_DIR + 'test_data_srilm_wiki_binning.csv')


'''
Load training and test data
'''
def load_training_test_data(file1, file2):
    file_data_1 = np.loadtxt(DATA_DIR + file1 , delimiter=',', dtype=str)
    file_data_2 = np.loadtxt(DATA_DIR + file2, delimiter=',', dtype=str)

    data_length = 1208
    file_data_1 = file_data_1[:data_length]
    file_data_2 = file_data_2[:data_length]

    count_of_test_data_from_each_file = int(len(file_data_1) / 10)
    test_data1 = file_data_1[:count_of_test_data_from_each_file]
    test_data2 = file_data_2[:count_of_test_data_from_each_file]
    test_data = np.concatenate((test_data1, test_data2), axis=0)

    head = file_data_1[:1]
    train_data1 = file_data_1[count_of_test_data_from_each_file:]
    train_data2 = file_data_2[count_of_test_data_from_each_file:]
    train_data = np.concatenate((train_data1, train_data2), axis=0)
    train_data = np.concatenate((head, train_data), axis=0)
    global data_obj
    data_obj = Data(data = train_data)
    global test_obj
    test_obj = Data(data = test_data)
    pass



'''
 calculate accuracy
'''
def calculate_accuracy(row, root):
    node_element = root
    if node_element.children == {}:
        return node_element.attr_name
    else:
        index_of_attribute = test_obj.get_column_index(node_element.attr_name)
        attribute_value_in_row = row[index_of_attribute]
        # check if attribute value(branch) in the row corresponds to node and then call recursively on the same node
        if attribute_value_in_row in node_element.children:
            child = node_element.children[attribute_value_in_row]
            return calculate_accuracy(row, child)
        else:
            leaf_nodes = find_common_label_generalization(node_element, [])
            unique_label, count = np.unique(leaf_nodes, return_counts=True)
            label_dict = dict(zip(unique_label, count))
            return max(label_dict, key=label_dict.get)

def find_common_label_generalization(node_element, leaf_nodes):
    for key, child in node_element.children.items():
        find_common_label_generalization(child, leaf_nodes)
    if node_element.children == {}:
        leaf_nodes.append(node_element.attr_name)
    return leaf_nodes

'''
find accuracy
'''
def find_accuracy(root, test_data):
    total_data = len(test_data.raw_data)
    match_count = 0
    for index in range(len(test_data.raw_data)):
        calculated_label = calculate_accuracy(test_data.raw_data[index], root)
        if calculated_label == test_data.get_column('label')[index]:
            match_count += 1
    accuracy = np.divide(float(match_count),float(total_data))*100
    #accuracy = (match_count/total_data)*100
    return accuracy


'''
Actual ID3 Implementation:
'''
def id3(data_set, attributes, label):

    if check_for_same_label_value(data_set.get_column('label')):
        return Tree(data_set.get_column('label')[0])
    else:
        max_info_gain_attribute = info_gain(data_set, attributes, data_set.get_column('label'))
        max_info_gain_attribute = str(max_info_gain_attribute)
        root_node = Tree(max_info_gain_attribute)
        if root_node.attr_name == '':
            pass
        node_attribute = data_set.attributes[root_node.attr_name].possible_vals
        for value in node_attribute:
            # add a child with branch value
            # get the sub data set of the attribute value
            sub_set_data = data_set.get_row_subset(max_info_gain_attribute, value)
            if len(sub_set_data) == 0:
                # find the most common label in the data_set for generalization
                unique_label, count = np.unique(data_set.get_column('label'), return_counts=True)
                # combining to form a dictionary
                label_dict = dict(zip(unique_label, count))
                common_label = max(label_dict, key=label_dict.get)
                root_node.children[value] = Tree(common_label)
            else:
                attr_value = ''
                if max_info_gain_attribute in attributes:
                    attr_value = attributes[max_info_gain_attribute]
                    print max_info_gain_attribute
                    del attributes[max_info_gain_attribute]
                if attributes == {}:
                    unique_label, count = np.unique(sub_set_data.get_column('label'), return_counts=True)
                    # combining to form a dictionary
                    label_dict = dict(zip(unique_label, count))
                    common_label = max(label_dict, key=label_dict.get)
                    root_node.children[value] = Tree(common_label)
                else:
                    root_node.children[value] = id3(sub_set_data, attributes, label)
                attributes[max_info_gain_attribute] = attr_value
    return root_node

'''
Returns tree depth. root node depth = 0
'''
def find_tree_depth(node, depth):
    if node.children == {}:
        return depth
    max_child_depth = 0
    for key, child in node.children.items():
        child_depth = find_tree_depth(child, depth+1)
        if max_child_depth < child_depth:
            max_child_depth = child_depth
    return max_child_depth

'''
Find Information gain
Parameters
----------
data_set: shrinked data on which you want to work
attributes: only thouse attributes that you want to work on
label_list: label of the shrinked data
'''
def info_gain(data_set, attributes, label_list):
    label_entropy = entropy(label_list)
    max_info_gain_attr = ''
    max_info_gain = 0
    data_set_length = np.size(label_list)
    for attribute, value in attributes.items():
        child_nodes_avg_entropy = 0
        attribute_possible_values = data_set.attributes[attribute].possible_vals
        for each_attr_value in attribute_possible_values:

            data_subset = data_set.get_row_subset(attribute, each_attr_value)
            data_subset.label = data_subset.get_column('label')
            attr_value_entropy = entropy(data_subset.label)
            attr_count = np.size(data_subset.label)
            child_nodes_avg_entropy += (attr_count/data_set_length)*attr_value_entropy
        attribute_gain = label_entropy - child_nodes_avg_entropy
        if max_info_gain <= attribute_gain:
            max_info_gain = attribute_gain
            max_info_gain_attr = attribute
    return max_info_gain_attr

'''
returns the calculated entropy 
parameters: shortlisted Label
'''
def entropy(label_list):
    # unique: list of unique possible values
    # count: count of each unique possible value
    unique, count = np.unique(label_list, return_counts=True)
    # combining to form a dictionary
    label_dict = dict(zip(unique, count))
    #length of the entire list
    label_list_length = np.size(label_list)
    entropy = 0
    for label,value in label_dict.items():
        # calculating entropy
        p_x = float(value) / float(label_list_length)
        entropy += - p_x * np.log2(p_x+ math.exp(-30))
    return entropy

'''
checkForSameLabelValue checks if all the labels are same
If same, return True else return False
'''
def check_for_same_label_value(label):
    if np.unique(label).size == 1:
        return True
    else:
        return False

'''
returns the data of a file
'''
def get_file_data (directory_path, file):
    file_data = np.loadtxt(directory_path + file, delimiter=',', dtype=str)
    file_data_obj = Data(data=file_data)
    return file_data_obj

'''
Implementation of ID3 by limiting the depth
Returns the root of the generated tree 
'''
def id3_for_cross_validation(data_set, attributes, label, depth, current_depth):

    if check_for_same_label_value(data_set.get_column('label')):
        current_depth += 1
        return Tree(data_set.get_column('label')[0])
    else:
        max_info_gain_attribute = info_gain(data_set, attributes, data_set.get_column('label'))
        root_node = Tree(max_info_gain_attribute)
        current_depth += 1
        node_attribute = data_set.attributes[root_node.attr_name].possible_vals
        for value in node_attribute:
            # add a child with branch value
            # get the sub data set of the attribute value
            sub_set_data = data_set.get_row_subset(max_info_gain_attribute, value)
            if len(sub_set_data) == 0:
                # find the most common label in the data_set for generalization
                unique_label, count = np.unique(data_set.get_column('label'), return_counts=True)
                # combining to form a dictionary
                label_dict = dict(zip(unique_label, count))
                common_label = max(label_dict, key=label_dict.get)
                root_node.children[value] = Tree(common_label)
            else:
                if depth == current_depth:
                    unique_label, count = np.unique(sub_set_data.get_column('label'), return_counts=True)
                    # combining to form a dictionary
                    label_dict = dict(zip(unique_label, count))
                    common_label = max(label_dict, key=label_dict.get)
                    root_node.children[value] = Tree(common_label)
                else:
                    attr_value = ''
                    if max_info_gain_attribute in attributes:
                        print max_info_gain_attribute
                        attr_value = attributes[max_info_gain_attribute]
                        del attributes[max_info_gain_attribute]
                    root_node.children[value] = id3_for_cross_validation(sub_set_data, attributes ,label, depth, current_depth)
                    attributes[max_info_gain_attribute] = attr_value

    return root_node


'''
Print Tree
'''
def print_tree(node):
    for key in node.children.keys():
        print(node.attr_name + "->" + key + ", " + node.children[key].attr_name)
        print_tree(node.children[key])

'''
Create a Tree node
'''
class Tree:
    def __init__(self, attr_name):
        self.attr_name = attr_name
        self.children = {}


'''
Picks m bootstrap samples which is used to create a classifier
'''
def pick_m_bootstrap_samples(m):
    data = np.loadtxt(DATA_DIR + training_file , delimiter=',', dtype=str)

    heading = data[0].copy()
    #bootstrap_sample = np.array([heading])
    bootstrap_sample = [heading]
    np.random.shuffle(data)
    data_len = len(data)-1
    for sample_number in range(m):
        index = random.randint(0, data_len)
        row = data[index].copy()
        bootstrap_sample.append(row)
        # handle a case where the random index picked is 0
    bootstrap_sample = np.asarray(bootstrap_sample)
    data_obj = Data(data=bootstrap_sample)
    return data_obj


def bagging_predict_on_test_data(root, test_data, recorded_labels):
    #total_data = len(test_data.raw_data)
    #match_count = 0
    for index in range(0, len(test_data.raw_data)):
        calculated_label = calculate_accuracy(test_data.raw_data[index], root)
        recorded_labels[index+1].append(calculated_label)

'''
Generates t number of classifiers by sampling m examples from the test set.
Finds the accuracy when a new classifier is created and records it
Cross validation done for number of classifiers from 1 to t
'''
def bagging_using_decision_trees(m, t, expected_test_data_labels):
    # dictionary of array where the array contains all the predicted labels for a perticular test example
    recorded_labels = {}
    for i in range(1, len(test_obj)+1):
        recorded_labels[i] = []

    for i in range(t):
        print("Started round", i)
        data_obj = pick_m_bootstrap_samples(m)
        #root = id3(data_obj, data_obj.attributes, data_obj.get_column('label'))
        root = id3_for_cross_validation(data_obj, data_obj.attributes, data_obj.get_column('label'), height, 0)
        bagging_predict_on_test_data(root, test_obj, recorded_labels)

        accuracy = find_accuracy_of_majority_label(expected_test_data_labels, recorded_labels)
        print("After round", i, " = ", accuracy)

    return recorded_labels

'''
Predicts the final label by picking the majority label generated by the classifiers
Returns the accuracy by comparing the predicted label and actual label  
'''
def find_accuracy_of_majority_label(expected_test_data_label, recorded_labels):
    match_count = 0
    for i in range(len(expected_test_data_label)):
        unique_label, count = np.unique(recorded_labels[i+1], return_counts=True)
        label_dict = dict(zip(unique_label, count))
        label = max(label_dict, key=label_dict.get)
        if label == expected_test_data_label[i]:
            match_count += 1
    accuracy = np.divide(float(match_count), float(len(expected_test_data_label)))*100
    return accuracy


def main():

    m = 870  # boostrap sample 40% percent of total train set
    t = 10  # number of classifiers to be found
    print("Height : ", height)
    expected_test_data_label = test_obj.get_column('label')
    recorded_labels = bagging_using_decision_trees(m, t, expected_test_data_label)

    accuracy = find_accuracy_of_majority_label(expected_test_data_label, recorded_labels)
    print("Accuracy on bagging using decision tree is: ", accuracy)


if __name__ == "__main__":
    main()
