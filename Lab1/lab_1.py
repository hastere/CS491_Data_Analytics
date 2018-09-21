import csv
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.metrics import jaccard_similarity_score
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import csv


def similarity(rowA, rowB):
    bin1 = []
    etc1 = []
    bin2 = []
    etc2 = []
    catCount = 0
    for key in rowA.keys():
        if str(key[-3:]) == "cat":
            if rowA[key] == rowB[key]:
                catCount += 1
        elif str(key[-3:]) == "bin":
            bin1.append(rowA[key])
            bin2.append(rowB[key])
        elif key != 'id' or key != 'target':
            etc1.append(rowA[key])
            etc2.append(rowB[key])
    catSim = catCount / len(rowA)
    binSim = jaccard_similarity_score(bin1, bin2)
    #shit???
    etcSim = 1 - spatial.distance.cosine(etc1, etc2)
    return (catSim + binSim + etcSim)/3

def main():
    #load data
    data = pd.read_csv('train.csv')
    #set cols to cat?
    for column in data.columns:
        if str(column[-3:]) == "cat":
            data[column] = data[column].astype('category')

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

    print "printing each nominal feature and their counts\n"
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
    rowA = data.iloc[1]
    rowB = data.iloc[2]
    print "\tsimilarity between rows 1 and 2 is: " + str(similarity(rowA, rowB)) + "\n"

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
    print "\n\tin summary " + str(counter) + " features are missing values\n"

    #answer to question 5
    #Please fill in the missing values, and briefly describe your approach.

    #we will be replacing the missing values with the mean or in the instance of categorical data,
    #mode of their respective column. first, we will replace the -1 values with Nan
    #value in order to take advantage of the pandas function that replaces NaN
    #values. Then, we will compute the mean of the column, rounding it in the
    #event that it is a piece of catagorical data

    print "question 5:\n"
    print "\tprinting column ps_ind_02_cat with missing values prior to replacement\n"
    print data['ps_ind_02_cat'].value_counts()
    print ""

    data = data.replace(-1,np.NaN)

    for column in data.columns:
        if column[-3:] == 'cat':
            data[column] = data[column].fillna(int(data[column].mode()))
        else:
            data[column] = data[column].fillna(float(data[column].mean()))
    print "\tprinting column with missing values after replacement\n"
    print data['ps_ind_02_cat'].value_counts()
    print ""

    #answer to question 6
    #How many classes are there in our target column?
    #Is our class balanced or highly imbalanced?
    #What challenge do you expect in classification task based on your observation?

    #there are 2 classes in the target column, and they are highly imbalanced
    print "question 6:\n"
    print "\tthere are two classes (1,0) that are highly imbalanced, as shown"
    print "\tin the frequencies below\n"
    print data['target'].value_counts(normalize=True)
    print "\n\tthe challenge here is that regular ml processing has a heavy bias"
    print "\ttowards classes that have high frequencies, usually only predicting"
    print "\tthat class. Minority classes will be treated as noise and filtered out.\n"

    #answer to question 7
    #Suppose we need to reduce the feature dimension to m
    #(m is a parameter such as 10),
    #and decided to use Principle Component Analysis (PCA) to do that.
    #Can you directly run PCA on our data and why?

    #If not, please preprocess the data and then run PCA with m = 10.
    #In your results, please provide the 10 principle components (vectors)
    #by a decreasing order, as well as data with reduced dimension.

    print "question 7:\n"

    pca = PCA(n_components=10)
    pca.fit(data)

    print "\n\t\t\tpca\n"
    print pd.DataFrame(pca.transform(data), columns=['PCA%i' % i for i in range(10)], index=data.index)

    # Code goes over here.
    return 0

if __name__ == "__main__":
    main()
