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
            counter += 1

    print "\tthere are " + str(counter) + " nominal features\n"

    print "printing counts of each catagory and their column name\n"
    for column in data.columns:
        if str(column[-3:]) == "cat" or str(column[-3:]) == "bin":
            print data[column].value_counts()
            print "\n"




    #answer to question 3
    #How would you compute similarity between feature vectors of any pair of samples (rows)?
    #Please first describe the steps or formula you will use to compute similarities.
    #We require that the similarity measure should have values between 0 and 1.
    #Then implement your function in script that takes two row indices as input
    #and return a similarity value as output.
    #(We require that your function  should be within 20 lines of codes)

    #we would use cosine similarity on the explanatory features
    #first, take the dot product of the two vectors, and then divide by the
    #product of their magnitude

    print "question 3:\n"

    #answer to question 4
    #How many features contain missing values?
    #For each feature containing missing values, what is the ratio of rows
    #(samples) that miss values? How many samples in total contain missing values?

    #13 features are missing values. See code below for frequencies and specific
    #column names
    counter = 0
    print "question 4:\n"
    print "\tratio of rows missing values for each feature with missing values\n"
    print "\t\tFeature\t\t\tRatio"
    for column in data.columns:
        temp = data[column].value_counts(normalize=True)
        if -1 in list(temp.keys()):
            print "\t\t" + str(column) + "\t" + str(temp[-1])
            counter += 1
    print "\n\tin summation " + str(counter) + " features are missing values"
    # Code goes over here.
    return 0

if __name__ == "__main__":
    main()
