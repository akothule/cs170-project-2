import numpy as np
import pandas as pd


def load_data(file_name):
    # load data from file
    data = np.loadtxt(file_name)

    # print shape of data frame
    #print(data.shape)
    # first column contains labels
    labels = data[:, 0]
    #print(labels)

    # remaining columns are features
    features = data[:, 1:]
    #print(features)

    return labels, features


def nearest_neighbor(labels, features):

    return 0


def forward_selection(labels, features):
    return 0

def backward_elimination(labels, features):
    return 0

def main():
    small_data = 'data/CS170_Small_Data__85.txt'
    large_data = 'data/CS170_Large_Data__67.txt'
    labels, features = 0, 0

    # choose a dataset to load
    while True:
        print("1. Small Dataset\n2. Large Dataset")
        # ask the user which dataset they want to load
        choice = input("Enter the number next to the corresponding dataset you want to load: ")

        # load the corresponding dataset
        if choice == '1':
            labels, features = load_data(small_data)
            print(f"\nSmall dataset loaded successfully. Dataset contains {features.shape[0]} rows "
                  f"and {features.shape[1]} features.")
            break
        elif choice == '2':
            labels, features = load_data(large_data)
            print(f"\nLarge dataset loaded successfully. Dataset contains {features.shape[0]} rows "
                  f"and {features.shape[1]} features.")
            break
        else:
            # ask again for user input if invalid
            print("Invalid Input. Please enter 1 or 2 depending on which dataset you want to load.\n")

    # choose which kind of search. forward or backward
    while True:
        print("1. Forward Selection\n2. Backward Elimination")
        # ask user which feature search they want to use
        choice = input("Enter the number next to the corresponding feature search you want to use: ")

        # execute corresponding search
        if choice == '1':
            forward_selection(labels, features)
            break
        elif choice == '2':
            backward_elimination(labels, features)
            break
        else:
            # ask again for user input if invalid
            print("Invalid Input. Please enter 1 or 2 depending on which search you want to use.\n")



    return 0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
