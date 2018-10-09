import csv
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.metrics import jaccard_similarity_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from scipy.spatial import distance
from scipy.stats.stats import pearsonr


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
    catSim = catCount
    binSim = jaccard_similarity_score(bin1, bin2) * len(bin1)
    etcSim = (1 - spatial.distance.cosine(etc1, etc2)) * len(etc1)
    return (catSim + binSim + etcSim)/57

def main():
    #load data
    data = pd.read_csv('train.csv')
    #set cols to cat?
    for column in data.columns:
        if str(column[-3:]) == "cat":
            data[column] = data[column].astype('category')

    #result = np.array(x).astype("float")

    #Answer to question 1
    #There are 57 explanatory features, with one being a target value indicating
    #that a claim was file and ID indication an individual client
    #ID cannot be considered as an explanatory feature as it is only a unique
    #identifier for the row.
    #code below
    print "question 1:\n"
    print "\tthere are " + str(len(data.columns) - 2) + " explanatory features\n"


    #answer to question 2
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
    #we would use cosine similarity on the explanatory features
    #first, take the dot product of the two vectors, and then divide by the
    #product of their magnitude

    print "question 3:\n"
    rowA = data.iloc[1]
    rowB = data.iloc[2]
    print "\tsimilarity between rows 1 and 2 is: " + str(similarity(rowA, rowB)) + "\n"

    #answer to question 4
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
    #there are 2 classes in the target column, and they are highly imbalanced
    print "question 6:\n"
    print "\tthere are two classes (1,0) that are highly imbalanced, as shown"
    print "\tin the frequencies below\n"
    print data['target'].value_counts(normalize=True)
    print "\n\tthe challenge here is that regular ml processing has a heavy bias"
    print "\ttowards classes that have high frequencies, usually only predicting"
    print "\tthat class. Minority classes will be treated as noise and filtered out.\n"

    #answer to question 7
    #We cannot run PCA on the dataset as-is, due to catagorical features existing.
    #These catagorical features must be encoded into one-hot tables and replaced in
    #the data set before running PCA.

    print "question 7:\n"

    pca = PCA(n_components=10)
    pcaData = pd.DataFrame()

    #one-thot the cat features
    for column in data.columns:
        if column[-3:] == 'cat':
            lb_style = LabelBinarizer()
            lb_results = lb_style.fit_transform(data[column])
            lb_table = pd.DataFrame(lb_results)
            lb_table.rename(columns=lambda x: column + "-" + str(x), inplace=True)
            #pcaData = pd.concat([pcaData,lb_table])
            #print pd.DataFrame(lb_table)
            pcaData = pcaData.join(lb_table)
        else:
            pcaData[column] = data[column]
    #run pca
    x = pcaData.drop(pcaData.columns[[0,1]], axis=1)
    pca.fit(x)
    cov_matrix = pca.get_covariance()
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    print "Top 10 principal components:"
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    counter = 0
    print eig_pairs[:10]
    print "\nResult of PCA\n"
    finalDataFrame = pd.DataFrame()
    finalDataFrame['id'] = pcaData['id']
    finalDataFrame['target'] = pcaData['target']
    finalDataFrame = finalDataFrame.join(pd.DataFrame(pca.transform(x), columns=['PCA%i' % i for i in range(10)], index=x.index))
    print finalDataFrame
    # Code goes over here.
    return 0


if __name__ == "__main__":
    main()
