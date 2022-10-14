"""
This module consists of functions performing mathematical operations.
"""
import math 
import random
from csv import reader
import sys

def euclidean_distance(row1: tuple, row2: tuple):
    """ Calculate the Euclidean distance between two vectors"""
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)


def get_neighbors(train: tuple, test_row: tuple, num_neighbors: int):
    """ Locates the most similar neighbors by calculcating distance between all
    dataset points and new data point.
    
    train - current dataset with coordinates
    test_row - a new data point 
    num_neighbors - the amount of neighbors that determine where to assign the test_row
    """
    distances = list()
    
    # Iterate through all the data rows in training dataset
    # and calculate the distance between the new data point (test_row) and currently iterated row.
    # Then save currently iterated row and it's distance to the new data point.
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    
    # Sort training rows by distances from shortest to longest
    distances.sort(key=lambda tup: tup[1])
    
    # Based on num_neighbors put the closest points in a neighbors list
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    
    return neighbors

def predict_classification(train, test_row, num_neighbors):
    """
    Classify to which class does the test_row belong.
    Collect all of the closest neighbors and make a prediction based on which class
    neighbors are more frequent.
    """
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def k_nearest_neighbors(train: list, test: list, num_neighbors: int):
    predictions = []
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions

def load_csv(filename: str):
    dataset = []
    with open(filename, 'r') as fh:
        csv_reader = reader(fh)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float(dataset: list, column: int):
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(dataset: list, column: int):
    """Convert 'class' column to integer values"""
    
    # Create a list of all values in given column
    class_values = [row[column] for row in dataset]
    # Make sure there are no repeated values
    unique = set(class_values)
    lookup = dict()
    
    # Make the unique values keys for the dictionary
    # and set their Set index as value for given key
    for i,value in enumerate(unique):
        lookup[value] = i
    
    # Convert class names to integer indexes
    for row in dataset:
        row[column] = lookup[row[column]]
    
    return lookup

def dataset_minmax(dataset: list):
    """Find min and max values for each column"""
    minmax = list()
    for i in range(len(dataset[0])):
        # Save all the values for current column
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        
        # Save min and max values for current column
        minmax.append([value_min, value_max])
    return minmax

def normalize_dataset(dataset: list, minmax: list):
    """
    Rescale dataset columns to range between 0 and 1.
    minmax is a list of minimum and maximum values for each column in the dataset.
    """
    for row in dataset:
        for i in range(len(row)):
            # Edit rows wit the equation
            # Row = (Row - min) / (max - min)
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def cross_validation_split(dataset: list, n_folds: int):
    """Split dataset into k folds"""
    dataset_split = []
    dataset_copy = dataset.copy()
    fold_size = int(len(dataset) / n_folds)
    
    for _ in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            # Pick a random number from 0 to dataset_copy list size
            index = random.randrange(len(dataset_copy))
            # Take out the row from dataset copy and put it into current fold
            fold.append(dataset_copy.pop(index))
        
        # Append created fold to dataset split list
        dataset_split.append(fold)
    
    # Return dataset splitted into n folds
    return dataset_split

def accuracy_metrics(actual: list, predicted: list):
    correct = 0
    
    # Iterate through predictions and count correct ones
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    
    # Return percentage of correct predictions
    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset: list, algorithm, n_folds: int, *args):
    """
    Evaluate an algorithm using a cross validation split
    """
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    
    for fold in folds:
        # Create a copy of folds and remove one fold for current iteration
        train_set = folds.copy()
        train_set.remove(fold)
        
        # Flatten the train_set - make it one dimension list
        train_set = sum(train_set, [])
        test_set = []
        
        # Go thorugh removed fold columns
        # and add them to a testset list
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            # Set the class column to None so that we could predict the class
            row_copy[-1] = None
        
        # Collect predictions and actual values
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        
        # Measure the final prediction accuracy
        accuracy = accuracy_metrics(actual, predicted)
        scores.append(accuracy)
    
    return scores

def test_knn(filename: str):
    # Make sure that generated numbers are the same for each run
    random.seed(1)
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    
    str_column_to_int(dataset, len(dataset[0])-1)
    
    n_folds = 5
    num_neighbors = 9
    
    scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


if __name__ == '__main__':
    print(load_csv(sys.argv[1])[0])