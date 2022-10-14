import random
from calculation_functions import str_column_to_float, str_column_to_int, predict_classification


def generate_dogs_dataset():
    """
    Generate a dog dataset with the following assumptions:
    
    Border collie
    weight 12 - 20kg
    height 46 - 56cm
    age Max 17 year
    
    Nowofounland
    weight 55 - 80kg
    height 63 - 71cm
    age Max 10 years
    """
    
    collies = [
        [
            round(random.uniform(12, 20), 2),
            round(random.uniform(46, 56), 2),
            round(random.uniform(2, 17), 2),
            "Border collie"
        ] for _ in range(100)
    ]
    
    newfounlands = [
        [
            round(random.uniform(55, 80), 2),
            round(random.uniform(63, 71), 2),
            round(random.uniform(2, 10), 2), 
            "Newfounland"
        ] for _ in range(100)
    ]
    
    return collies + newfounlands


def predict_dogs(weight: float, height: float, age: float):
    """
    Show predictions for a dog with given physical parameters.
    """
    dataset = generate_dogs_dataset()
    
    # Convert class column (the last column) to integer values
    str_column_to_int(dataset, len(dataset[0])-1)
    
    # Define model parameter and new record to be added
    num_neighbors = 5
    row = [weight, height, age]
    # Predict the label
    label = predict_classification(dataset, row, num_neighbors)
    print('Data=%s, Predicted: %s' % (row, label))
    
if __name__ == "__main__":
    predict_dogs(weight=82, height=55, age=5)