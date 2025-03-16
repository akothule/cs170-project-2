import sys
import time
import numpy as np
import pandas as pd
import logging


# initialize logger
def init_logger():
    # initialize logger
    _logger = logging.getLogger("CS_170_Project_2")
    # set log level to info
    _logger.setLevel(logging.INFO)
    # create a file handler
    file_handler = logging.FileHandler("cs_170_project_2.log")
    # create a console logger
    console_handler = logging.StreamHandler(sys.stdout)
    # add handlers to logger
    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)

    return _logger


# initialize a global variable logger
logger = init_logger()


def load_data(file_name):
    # load data from file
    data = np.loadtxt(file_name)

    # print shape of data frame
    # first column contains labels
    labels = data[:, 0]

    # remaining columns are features
    features = data[:, 1:]

    return labels, features


# calculate Euclidean distance between 2 points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def leave_one_out_cross_validation(labels, features, current_set_of_features, feature_considered):
    # make a copy of features
    features_with_current_set = features.copy()
    # shape of features array
    num_of_instances = features.shape[0]
    num_of_features = features.shape[1]

    if feature_considered not in current_set_of_features:
        # traverse features, setting non-used features to 0
        # forward selection
        for i in range(num_of_features):
            if i not in current_set_of_features and i != feature_considered:
                features_with_current_set[:, i] = 0
    else:
        # traverse features, setting non-used features to 0
        # backward elimination
        # set feature_considered column to 0, as if you're removing it from current_set_of_features
        for i in range(num_of_features):
            if i not in current_set_of_features or i == feature_considered:
                features_with_current_set[:, i] = 0

    # initialize number of correctly classified object
    number_correctly_classified = 0

    # loop through all instances
    for i in range(num_of_instances):
        # store object to be classified
        object_to_classify = features_with_current_set[i]
        # get the label of the object
        label_object_to_classify = labels[i]

        # initialize distance and location to infinity
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        # initialize label of nearest neighbor
        nearest_neighbor_label = 0
        # loop through number of instances
        for k in range(num_of_instances):
            # don't consider itself as its nearest neighbor
            if k != i:
                # get the kth instance
                training_instance = features_with_current_set[k]
                # calculate distance between 2 objects
                distance = euclidean_distance(object_to_classify, training_instance)
                # keep track of minimum distance
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    # get the location and label of nearest neighbor
                    nearest_neighbor_location = k
                    nearest_neighbor_label = labels[nearest_neighbor_location]

        # if object is correctly classified, increment number_correctly_classified
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1

    # calculate accuracy by dividing number correctly classified by total number of instances
    return number_correctly_classified / num_of_instances


def forward_selection(labels, features):
    # initialize empty set of current features
    current_set_of_features = []
    # initialize empty set of best overall features
    best_overall_set_of_features = []
    # overall accuracy of best overall features
    best_overall_accuracy = leave_one_out_cross_validation(labels, features, current_set_of_features, None)
    # number of features in dataset
    num_of_features = features.shape[1]

    # calculate accuracy of empty set of features
    logger.info(f"\nBeginning Forward Selection search")
    logger.info(
        f"Accuracy of current feature set {[f + 1 for f in current_set_of_features]}: {100 * best_overall_accuracy:.2f}%")

    # loop through features
    for i in range(num_of_features):
        logger.info(f"\nOn the {i + 1}th level of the search tree")
        # initialize variable to keep track of which feature to add at ith level
        feature_to_add_at_this_level = None
        # keep track of best accuracy at current level
        best_accuracy_at_current_level = 0

        # loop through features
        for k in range(num_of_features):
            # if k feature is not already in current set of features, consider adding it
            if k not in current_set_of_features:
                # measure accuracy if you added k
                accuracy = leave_one_out_cross_validation(labels, features, current_set_of_features, k)
                logger.info(f"--Considering adding the {k + 1} feature. Accuracy is {100 * accuracy:.2f}%")

                if accuracy > best_accuracy_at_current_level:
                    # update best accuracy at current level
                    best_accuracy_at_current_level = accuracy
                    # add k feature
                    feature_to_add_at_this_level = k

        # add k feature to current set of features
        current_set_of_features.append(feature_to_add_at_this_level)
        logger.info(f"On level {i + 1}, I added feature {feature_to_add_at_this_level + 1} to current set")
        logger.info(f"Accuracy of current feature set {[f + 1 for f in current_set_of_features]}: {100*best_accuracy_at_current_level:.2f}%")

        # track best overall feature set
        if best_accuracy_at_current_level > best_overall_accuracy:
            best_overall_accuracy = best_accuracy_at_current_level
            best_overall_set_of_features = current_set_of_features.copy()
        elif best_accuracy_at_current_level < best_overall_accuracy and i != num_of_features - 1:
            logger.info("WARNING: Accuracy has decreased. Continuing search in case of local maxima")

    logger.info(f"\nBest set of features: {[f + 1 for f in best_overall_set_of_features]}")
    logger.info(f"Overall accuracy: {100*best_overall_accuracy:.2f}%")

    return best_overall_set_of_features


def backward_elimination(labels, features):
    # number of features in dataset
    num_of_features = features.shape[1]
    # initialize full set of current features
    current_set_of_features = list(range(num_of_features))
    # initialize set of best overall features, copying current set
    best_overall_set_of_features = current_set_of_features.copy()
    # overall accuracy of all features
    best_overall_accuracy = leave_one_out_cross_validation(labels, features, current_set_of_features, None)

    # calculate accuracy of full set of features
    logger.info(f"\nBeginning Backward Elimination search")
    logger.info(f"Accuracy of current feature set {[f + 1 for f in current_set_of_features]}: {100 * best_overall_accuracy:.2f}%")

    # loop backwards through search tree
    for i in range(num_of_features, 0, -1):
        logger.info(f"\nOn the {i}th level of the search tree")
        # initialize variable to keep track of which feature to remove at ith level
        feature_to_remove_at_this_level = None
        # keep track of best accuracy at current level
        best_accuracy_at_current_level = 0

        # loop through features
        for k in range(num_of_features):
            # if k feature is already in current set of features, consider removing it
            if k in current_set_of_features:
                # measure accuracy if you removed k
                accuracy = leave_one_out_cross_validation(labels, features, current_set_of_features, k)
                logger.info(f"--Considering removing the {k + 1} feature. Accuracy is {100 * accuracy:.2f}%")

                if accuracy > best_accuracy_at_current_level:
                    # update best accuracy at current level
                    best_accuracy_at_current_level = accuracy
                    # remove k feature
                    feature_to_remove_at_this_level = k

        # add k feature to current set of features
        current_set_of_features.remove(feature_to_remove_at_this_level)
        logger.info(f"On level {i}, I removed feature {feature_to_remove_at_this_level + 1} from current set")
        logger.info(f"Accuracy of current feature set {[f + 1 for f in current_set_of_features]}: {100 * best_accuracy_at_current_level:.2f}%")

        # track best overall feature set
        if best_accuracy_at_current_level > best_overall_accuracy:
            best_overall_accuracy = best_accuracy_at_current_level
            best_overall_set_of_features = current_set_of_features.copy()
        elif best_accuracy_at_current_level < best_overall_accuracy and i != 0:
            logger.info("WARNING: Accuracy has decreased. Continuing search in case of local maxima")

    logger.info(f"\nBest set of features: {[f + 1 for f in best_overall_set_of_features]}")
    logger.info(f"Overall accuracy: {100 * best_overall_accuracy:.2f}%")

    return best_overall_set_of_features


def main():
    small_data = 'data/CS170_Small_Data__85.txt'
    large_data = 'data/CS170_Large_Data__67.txt'
    labels, features = None, None

    # choose a dataset to load
    while True:
        logger.info("1. Small Dataset\n2. Large Dataset")
        # ask the user which dataset they want to load
        choice = input("Enter the number next to the corresponding dataset you want to load: ")

        # load the corresponding dataset
        if choice == '1':
            labels, features = load_data(small_data)
            logger.info(f"\nSmall dataset loaded successfully. Dataset contains {features.shape[0]} instances "
                  f"and {features.shape[1]} features (not including the class attribute).")
            break
        elif choice == '2':
            labels, features = load_data(large_data)
            logger.info(f"\nLarge dataset loaded successfully. Dataset contains {features.shape[0]} instances "
                  f"and {features.shape[1]} features (not including the class attribute).")
            break
        else:
            # ask again for user input if invalid
            logger.info("Invalid Input. Please enter 1 or 2 depending on which dataset you want to load.\n")

    # time stamp for calculating duration of runtime
    start_time = time.time()

    # choose which kind of search. forward or backward
    while True:
        logger.info("1. Forward Selection\n2. Backward Elimination")
        # ask user which feature search they want to use
        choice = input("Enter the number next to the corresponding feature search you want to use: ")

        start_time = time.time()
        # execute corresponding search
        if choice == '1':
            forward_selection(labels, features)
            break
        elif choice == '2':
            backward_elimination(labels, features)
            break
        else:
            # ask again for user input if invalid
            logger.info("Invalid Input. Please enter 1 or 2 depending on which search you want to use.\n")

    end_time = time.time()
    logger.info(f"Total runtime = {end_time - start_time:.2f} seconds")

    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
