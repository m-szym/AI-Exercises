import numpy as np

#from decision_tree import DecisionTree
#from random_forest_solution import RandomForest
from load_data import generate_data, load_titanic

from random_tree import DecisionTree
from random_forest import RandomForest


def main():
    np.random.seed(123)

    train_data, test_data = load_titanic()

    print("\n\tDecision Tree")
    dt = DecisionTree({"depth": 14})
    dt.train(*train_data)
    print("Training data", end=" ")
    dt.evaluate(*train_data)
    print("Test data", end=" ")
    dt.evaluate(*test_data)

    print("\n\tRandom Forest")
    rf = RandomForest({"ntrees": 10, "feature_subset": 2, "depth": 14})
    rf.train(*train_data)
    print("Training data", end=" ")
    rf.evaluate(*train_data)
    print("Test data", end=" ")
    rf.evaluate(*test_data)

if __name__=="__main__":
    main()