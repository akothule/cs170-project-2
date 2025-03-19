import sys
import time
import numpy as np
import pandas as pd
import logging

from chart import create_chart


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


def export_to_csv(accuracy_percentages, feature_set_labels, title, filename):
    # create a DataFrame with the data
    df = pd.DataFrame({
        'Feature Set': feature_set_labels,
        'Accuracy (%)': accuracy_percentages
    })

    # add a comment row with the title (Google Sheets will ignore this)
    with open(filename, 'w') as f:
        f.write(f"# {title}\n")

    # append the actual data to the CSV file
    df.to_csv(filename, index=False, mode='a')
    # logger.info(f"Data exported to {filename} successfully")


# calculate Euclidean distance between 2 points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def calculate_accuracy(actual):
    return actual * 100

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


def forward_selection(labels, features, dataset_type):
    # initialize empty set of current features
    current_set_of_features = []
    # initialize empty set of best overall features
    best_overall_set_of_features = []
    # overall accuracy of best overall features
    best_overall_accuracy = calculate_accuracy(leave_one_out_cross_validation(labels, features, current_set_of_features, None))
    # number of features in dataset
    num_of_features = features.shape[1]

    # store accuracy at each level for plotting
    all_accuracies = []
    feature_sets = []
    feature_set_labels = []
    all_accuracies.append(best_overall_accuracy)
    feature_sets.append([])
    feature_set_labels.append("{}") # used for plotting the chart

    # begin search
    logger.info(f"Running nearest neighbor with all {num_of_features} features, using \"leaving-one-out\" "
                f"evaluation, I get an accuracy of {calculate_accuracy(leave_one_out_cross_validation(labels, features, list(range(num_of_features)), None))}%")
    logger.info(f"Beginning search.")

    # loop through features
    for i in range(num_of_features):
        # initialize variable to keep track of which feature to add at ith level
        feature_to_add_at_this_level = None
        # keep track of best accuracy at current level
        best_accuracy_at_current_level = 0

        # loop through features
        for k in range(num_of_features):
            # if k feature is not already in current set of features, consider adding it
            if k not in current_set_of_features:
                # measure accuracy if you added k
                accuracy = calculate_accuracy(leave_one_out_cross_validation(labels, features, current_set_of_features, k))
                print_feature_set = {f + 1 for f in current_set_of_features}
                print_feature_set.add(k + 1)
                if len(print_feature_set) > 0:
                    logger.info(f" Using feature(s) {print_feature_set}. accuracy is {accuracy:.1f}%")
                else:
                    logger.info(f" Using feature(s) {'{}'}. accuracy is {accuracy:.1f}%")

                if accuracy > best_accuracy_at_current_level:
                    # update best accuracy at current level
                    best_accuracy_at_current_level = accuracy
                    # add k feature
                    feature_to_add_at_this_level = k

        # add k feature to current set of features
        current_set_of_features.append(feature_to_add_at_this_level)

        # track best overall feature set
        if best_accuracy_at_current_level > best_overall_accuracy:
            best_overall_accuracy = best_accuracy_at_current_level
            best_overall_set_of_features = current_set_of_features.copy()
        elif best_accuracy_at_current_level < best_overall_accuracy and i != num_of_features - 1 and i != 0:
            logger.info("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")

        if i < num_of_features - 1:
            print_best_set = {f + 1 for f in current_set_of_features}
            logger.info(f"Feature set {print_best_set} was the best, accuracy is {best_accuracy_at_current_level:.1f}%")

        # store for plotting
        all_accuracies.append(best_accuracy_at_current_level)
        feature_sets.append(current_set_of_features.copy())
        feature_set_labels.append("{" + ",".join(str(f+1) for f in current_set_of_features) + "}")

    print_best_set = {f + 1 for f in best_overall_set_of_features}
    logger.info(f"Finished search!! The best feature subset is {print_best_set}, which has an accuracy of {best_overall_accuracy:.1f}%")

    # export to csv file
    title = 'Accuracy of increasingly large subsets of features discovered by forward selection'
    xlabel = 'Current Feature Set: Forward Selection'
    ylabel = 'Accuracy (%)'
    csv_file = f'data/forward_selection_results_{dataset_type}.csv'
    chart_file = csv_file.replace('.csv', '_chart.png')
    export_to_csv(all_accuracies, feature_set_labels, title, csv_file)
    # export to chart
    create_chart(all_accuracies, feature_set_labels, title, xlabel, ylabel, chart_file)

    return best_overall_set_of_features


def backward_elimination(labels, features, dataset_type):
    # number of features in dataset
    num_of_features = features.shape[1]
    # initialize full set of current features
    current_set_of_features = list(range(num_of_features))
    # initialize set of best overall features, copying current set
    best_overall_set_of_features = current_set_of_features.copy()
    # overall accuracy of all features
    best_overall_accuracy = calculate_accuracy(leave_one_out_cross_validation(labels, features, current_set_of_features, None))

    # store accuracy at each level for plotting
    all_accuracies = []
    feature_sets = []
    feature_set_labels = []
    # store initial state for plotting
    all_accuracies.append(best_overall_accuracy)
    feature_sets.append(current_set_of_features.copy())
    feature_set_labels.append("{" + ",".join(str(f+1) for f in current_set_of_features) + "}")

    # begin search
    logger.info(f"Running nearest neighbor with no features, using \"leaving-one-out\" "
                f"evaluation, I get an accuracy of {calculate_accuracy(leave_one_out_cross_validation(labels, features, list(), None))}%")
    logger.info(f"Beginning search.")

    # loop backwards through search tree
    for i in range(num_of_features, 0, -1):
        # initialize variable to keep track of which feature to remove at ith level
        feature_to_remove_at_this_level = None
        # keep track of best accuracy at current level
        best_accuracy_at_current_level = 0

        # loop through features
        for k in range(num_of_features):
            # if k feature is already in current set of features, consider removing it
            if k in current_set_of_features:
                # measure accuracy if you removed k
                accuracy = calculate_accuracy(leave_one_out_cross_validation(labels, features, current_set_of_features, k))
                print_feature_set = {f + 1 for f in current_set_of_features}
                print_feature_set.remove(k + 1)
                if len(print_feature_set) > 0:
                    logger.info(f" Using feature(s) {print_feature_set}. accuracy is {accuracy:.1f}%")
                else:
                    logger.info(f" Using feature(s) {'{}'}. accuracy is {accuracy:.1f}%")

                if accuracy > best_accuracy_at_current_level:
                    # update best accuracy at current level
                    best_accuracy_at_current_level = accuracy
                    # remove k feature
                    feature_to_remove_at_this_level = k

        # add k feature to current set of features
        current_set_of_features.remove(feature_to_remove_at_this_level)

        # track best overall feature set
        if best_accuracy_at_current_level > best_overall_accuracy:
            best_overall_accuracy = best_accuracy_at_current_level
            best_overall_set_of_features = current_set_of_features.copy()
        elif best_accuracy_at_current_level < best_overall_accuracy and i != 1:
            logger.info("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")

        print_best_set = {f + 1 for f in current_set_of_features}
        if len(print_best_set) > 0:
            logger.info(f"Feature set {print_best_set} was the best, accuracy is {best_accuracy_at_current_level:.1f}%")

        # store for plotting
        all_accuracies.append(best_accuracy_at_current_level)
        feature_sets.append(current_set_of_features.copy())
        feature_set_labels.append("{" + ",".join(str(f+1) for f in current_set_of_features) + "}" if current_set_of_features else "{}")

    print_best_set = {f + 1 for f in best_overall_set_of_features}
    logger.info(
        f"Finished search!! The best feature subset is {print_best_set}, which has an accuracy of {best_overall_accuracy:.1f}%")

    # export to csv file
    title = 'Accuracy of increasingly small subsets of features discovered by backward elimination'
    xlabel = 'Current Feature Set: Backward Elimination'
    ylabel = 'Accuracy (%)'
    csv_file = f'data/backward_elimination_results_{dataset_type}.csv'
    chart_file = csv_file.replace('.csv', '_chart.png')
    export_to_csv(all_accuracies, feature_set_labels, title, csv_file)
    # export to chart
    create_chart(all_accuracies, feature_set_labels, title, xlabel, ylabel, chart_file)

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
            dataset_type = 'small'
            logger.info(f"Small dataset loaded successfully. Dataset contains {features.shape[0]} instances "
                  f"and {features.shape[1]} features (not including the class attribute).")
            break
        elif choice == '2':
            labels, features = load_data(large_data)
            dataset_type = 'large'
            logger.info(f"Large dataset loaded successfully. Dataset contains {features.shape[0]} instances "
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
            forward_selection(labels, features, dataset_type)
            break
        elif choice == '2':
            backward_elimination(labels, features, dataset_type)
            break
        else:
            # ask again for user input if invalid
            logger.info("Invalid Input. Please enter 1 or 2 depending on which search you want to use.\n")

    end_time = time.time()
    # logger.info(f"Total runtime = {end_time - start_time:.2f} seconds")

    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/