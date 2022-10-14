"""
K Nearest Neighbors algorithm implemented based on tutorial:
https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

Download this dataset into current directory: https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv
"""
import sys

from calculation_functions import load_csv, str_column_to_float, str_column_to_int, predict_classification, test_knn

def test_new_prediction(filename):
    # Load dataset to memory
    dataset = load_csv(filename)
    # Make sure that values in columns are in float format (except the class column)
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
        
    # Convert class column (the last column) to integer values
    str_column_to_int(dataset, len(dataset[0])-1)
    
    # Define model parameter and new record to be added
    num_neighbors = 5
    row = [5.7,2.9,4.2,1.3]
    
    # Predict the label
    label = predict_classification(dataset, row, num_neighbors)
    print('Data=%s, Predicted: %s' % (row, label))


if __name__ == "__main__":
    # Read a filename with database from the command line
    filename = sys.argv[1]
    
    # test_new_prediction(filename)
    test_knn(filename)
