import random
import math

def separate_by_class(dataset: list):
    """Return a dictionary with class names as keys and full rows as values."""
    separated = dict()
    
    # Iterate over the whole dataset
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        
        # If the class wasn't added to dictionary as a key yet
        if class_value not in separated:
            separated[class_value] = list()
            
        # Remember current dataset row as assigned to the class
        separated[class_value].append(vector)
    return separated

def mean(numbers: list[int]):
    """ Return mean value of a list of numbers"""
    return sum(numbers)/float(len(numbers))

def stddev(numbers: list):
    """Return standard deviation of a list of numbers"""
    average = mean(numbers)
    variance = sum([(x-average)**2 for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

def summarize_dataset(dataset: list):
    """Calculate the mean, standard deviation and count for each column in dataset"""
    summaries = [(mean(column), stddev(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries

def summarize_by_class(dataset: list):
    """Summarize data for particular classes for each of the columns"""
    
    separated = separate_by_class(dataset)
    summaries = dict()
    
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    
    return summaries


def calculate_probability(x, mean, stddev):
    exponent = math.exp(-((x-mean)**2 / (2*stddev**2)))
    return (1 / (math.sqrt(2*math.pi) * stddev)) * exponent


def calculate_class_probabilities(summaries: dict, row: list):
    # Count total rows which are stored in the 2nd column for each label
    total_rows = sum([summaries[label][0][2] for label in summaries])
    
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            # Calculate probability for each column value to be in proper range (Gaussian)
            # Then multiply the probabilities to get the final result
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities


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
            0
        ] for _ in range(100)
    ]
    
    newfounlands = [
        [
            round(random.uniform(55, 80), 2),
            round(random.uniform(63, 71), 2),
            round(random.uniform(2, 10), 2), 
            1
        ] for _ in range(100)
    ]
    
    return collies + newfounlands


if __name__ == '__main__':
    ds = generate_dogs_dataset()
    summaries = summarize_by_class(ds)
    first_dog = ds[120]
    
    probabilities = calculate_class_probabilities(summaries, first_dog)
    
    print(f"First dog: {first_dog}")
    print(30*'-')
    print(probabilities)