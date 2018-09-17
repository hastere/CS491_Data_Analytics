import csv
import numpy as np
import pandas as pd
from scipy import spatial

import csv


def main():
    #load data
    data = pd.read_csv('train.csv')
    #result = np.array(x).astype("float")

    #Answer to question 1
    #How many explanatory features are there in the training data (train.csv)?
    #Could id be considered as an explanatory feature and why?

    #There are 57 explanatory features, with one being a target value indicating
    #that a claim was file and ID indication an individual client
    #ID cannot be considered as an explanatory feature as it is only a unique
    #identifier for the row.
    #code below
    print "question 1:\n"
    print "\tthere are " + str(len(data.columns) - 2) + " explanatory features\n"


    #answer to question 2
    #Among the explanatory features in train.csv, how many are nominal (i.e., binary or categorical)?
    #Please list all nominal features respectively.
    #For each of them, please provide the count of each category.
    #There are 31 (postfix of bin or cat indicates what kind of feature this is)

    print "question 2:\n"

    counter = 0
    for column in data.columns:
        if str(column[-3:]) == "cat" or str(column[-3:]) == "bin":
            ++counter

    print "\tthere are " + str(counter) + " nominal features\n"

    print "printing counts of each catagory and their column name\n"
    for column in data.columns:
        if str(column[-3:]) == "cat" or str(column[-3:]) == "bin":
            print data[column].value_counts()
            print "\n"




    dataSetI = [3, 45, 7, 2]
    dataSetII = [2, 54, 13, 15]
    result = 1 - spatial.distance.cosine(dataSetI, dataSetII)

    # Code goes over here.
    return 0

if __name__ == "__main__":
    main()
