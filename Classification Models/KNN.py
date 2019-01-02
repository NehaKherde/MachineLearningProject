#####################################################################
# Code to find K - Nearest Neighbours.
# Used Euclidean distance to find nearest neighbours.
# Picked best K value by cross validating on different values of K
# Author: Neha Kherde
#####################################################################


from create_train_and_test import *
from create_train_and_test_maitrey import *


'''
Params: a point from test and train data
Returns euclidean distance between the 2 points
'''
def find_euclidean_distance(test_record, train_record):
    feature_length = len(test_record)-1
    sum = 0
    for index in range(feature_length):
        sum += (test_record[index] - train_record[index])**2
    return sum**0.5


'''
Predicts label,
by finding majority label among the k nearest neighbours
'''
def find_majority_label(distance_array):
    # 1.0
    positive_ans = 0
    # 0.0
    negative_ans = 0
    for each in distance_array:
        if each[1] == 1.0:
            positive_ans += 1
        else:
            negative_ans += 1
    if positive_ans > negative_ans:
        return 1.0
    else:
        return 0.0


'''
Implementation of KNN
Finds K nearest neighbours, predicts the label and returns the accuracy on test data. 
Here K is the best hyper parameter 
'''
def knn(test_data, train_data):
    distance_array = []
    matched_ans = 0
    #k_val = int(((len(train_data))**0.5)-1)
    k_val = 50
    for test_record in test_data:
        for train_record in train_data:
            label_index = len(test_record) - 1
            distance = find_euclidean_distance(test_record, train_record)
            distance_array.append((distance, train_record[label_index]))

        # After finding the distance of test_record with all train_data, pick k points that are nearest to test record.
        distance_array.sort(key=lambda tup: tup[0])
        distance_array = distance_array[:k_val]

        # Find majority label among the k nearest neighbours and assign that as predicted label
        expected_label = find_majority_label(distance_array)

        # check if predicted label on the test_record is same are original label and find the accuracy
        if expected_label == test_record[len(test_record) - 1]:
            matched_ans += 1
            #print("match found")
        else:
            pass
            #print("no match found")

    # Find accuracy:
    accuracy = (float(matched_ans)/float(len(test_data))) * 100

    print("Number of matched answers: " + str(matched_ans) + "/" +str(len(test_data)))
    print("Percentage accuracy : " + str(accuracy))


'''
Implementation of KNN using dynamically passed K value
Finds K nearest neighbours, predicts the label and returns the accuracy on test data. 
'''
def knn_cross_validation(test_data, train_data, k):
    distance_array = []
    matched_ans = 0
    k_val = k
    for test_record in test_data:
        for train_record in train_data:
            label_index = len(test_record) - 1
            distance = find_euclidean_distance(test_record, train_record)
            distance_array.append((distance, train_record[label_index]))

        # After finding the distance of test_record with all train_data, pick k points that are nearest to test record.
        distance_array.sort(key=lambda tup: tup[0])
        distance_array = distance_array[:k_val]

        # Find majority label among the k nearest neighbours and assign that as predicted label
        expected_label = find_majority_label(distance_array)

        # check if predicted label on the test_record is same are original label and find the accuracy
        if expected_label == test_record[len(test_record) - 1]:
            matched_ans += 1
            #print("match found")
        else:
            pass
            #print("no match found")

    # Find accuracy:
    accuracy = (float(matched_ans)/float(len(test_data))) * 100

    print("****Number of matched answers: " + str(matched_ans) + "/" +str(len(test_data)))
    print("****Percentage accuracy : " + str(accuracy))
    return accuracy, matched_ans


'''
Implementation of KNN using cross validation for different values of K = [10, 30, 35, 45, 50]
Finds K nearest neighbours, predicts the label and returns the accuracy on test data. 
Predicts the best hyper parameter 
'''
def cross_fold_validation():
    fake_corpus_file = "tagged_vectors_gen_dropshuff.npy"
    correct_corpus_file = "tagged_vectors_spacy.npy"
    k_array = [10, 30, 45, 50, 35]
    accuracy_arr = []
    matched_ans_count_arr = []
    for index in range(len(k_array)):
        print("Start of loop with k val : ", k_array[index])
        #test_data, train_data = get_test_train_data(correct_corpus_file, fake_corpus_file)
        test_data, train_data = getTrainTest(correct_corpus_file, fake_corpus_file)
        accuracy, matched_ans = knn_cross_validation(test_data, train_data, k_array[index])

        accuracy_arr.append(accuracy)
        matched_ans_count_arr.append(matched_ans)
        print("End of loop with k val : ", k_array[index])

    best_result_index = accuracy_arr.index(max(accuracy_arr))
    print("Accuracy arr: ", accuracy_arr)
    print("Matched answer count: ", matched_ans_count_arr)
    print("************ Crossfold best parameters: *************")
    print("Accuracy: ", accuracy_arr[best_result_index], " best hyper parameter: ", k_array[best_result_index], " matched answer count: ", matched_ans_count_arr[best_result_index])


def main():
    fake_corpus_file = "final_tagged_vectors_bilstm.npy"
    correct_corpus_file = "tagged_vectors_gen_bilstm.npy"
    #test_data, train_data = get_test_train_data(correct_corpus_file, fake_corpus_file)
    test_data, train_data = getTrainTest(correct_corpus_file, fake_corpus_file)
    knn(test_data, train_data)

   #cross_fold_validation()


if __name__ == "__main__": main()