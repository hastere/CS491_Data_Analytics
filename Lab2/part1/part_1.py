import csv
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer



def main():
    #load data
    trainInput = pd.read_csv('train.csv')
    testInput = pd.read_csv('test.csv')

    #need to preprocess via one-hot encoding of categorical variables.
    #unlike in r, sklearn's decision trees/random forests do not handle catagorical
    #data well
    train = pd.DataFrame()
    test = pd.DataFrame()


    for column in trainInput.columns:
        if column[-3:] == 'cat':
            lb_style = LabelBinarizer()
            lb_results_train = lb_style.fit_transform(trainInput[column])
            lb_results_test = lb_style.fit_transform(testInput[column])
            lb_table_train = pd.DataFrame(lb_results_train)
            lb_table_test = pd.DataFrame(lb_results_test)
            lb_table_train.rename(columns=lambda x: column + "-" + str(x), inplace=True)
            lb_table_test.rename(columns=lambda x: column + "-" + str(x), inplace=True)
            train = train.join(lb_table_train)
            test = test.join(lb_table_test)
        else:
            train[column] = trainInput[column]
            test[column] = testInput[column]
    train = train.fillna(0)
    test = test.fillna(0)

    trainTarget = train["target"]
    trainFeatureNames = list(train.columns[2:])
    trainFeatures = train[trainFeatureNames]
    testTarget = test["target"]
    testFeatureNames = list(test.columns[2:])
    testFeatures = test[testFeatureNames]
    #TODO catagorical columns should be converted to catagorical type
    dataTree = tree.DecisionTreeClassifier(min_samples_split = 20, max_depth = 15, random_state = 69)
    dataTree = dataTree.fit(trainFeatures, trainTarget)
    testPrediction = dataTree.predict(testFeatures)
    trainPrediction = dataTree.predict(trainFeatures)
    print "\nNow using decision trees"
    #Answer to question 1.i
    trainConfusion = confusion_matrix(trainTarget, trainPrediction)
    testConfusion = confusion_matrix(testTarget, testPrediction)
    print "\nTraining set confusion matrix"
    print trainConfusion
    print "\nTest set confusion matrix"
    print testConfusion
    #Answer to question 1.ii
    #TODO see about spliting scores out by class
    print "\nTraining set prediction, recal, and f-score"
    print precision_recall_fscore_support(trainTarget, trainPrediction, average='binary')[:3]
    #Answer to question 1.iii
    print "\nTest set precision, recall, and f-score"
    print precision_recall_fscore_support(testTarget, testPrediction, average='binary')[:3]

    #Answer to question 1.iv
    print "\nprecision and recall decrease between the training data and test data."
    print "this is because the decision tree was trained on the training data, which"
    print "resulted in a perfect fit to that specific set. this is an example of why"
    print "you shouldn't train a model on test data\n"

    #Answer to question 1.v
    print "\nfor the train data"
    x = (trainConfusion[1][1])
    y = (x + trainConfusion[1][0])
    print "\tTrue Positive Rate = " + str(float(x)/float(y))
    x = (trainConfusion[0][1])
    y = (x + trainConfusion[0][0])
    print "\tFalse Positive Rate = " + str(float(x)/float(y))
    #Answer to question 1.vi
    print "for the test data"
    x = (testConfusion[1][1])
    y = (x + testConfusion[1][0])
    print "\tTrue Positive Rate = " + str(float(x)/float(y))
    x = (testConfusion[0][1])
    y = (x + testConfusion[0][0])
    print "\tFalse Positive Rate = " + str(float(x)/float(y))

    #Ansrew to question 1.vii
    print "\nagain, testing on the training data results in perfect classification"
    print "of training data, as the model is based off of the test data. The test"
    print "data is different, which exposes some errors in prediction\n"

    #Answer to question 1.viii
    print "Due to variance in samples the recall value is a better way of evaluating"
    print "performance"

    print "\nNow using RandomForestClassifier as a model"
    randomForest =  RandomForestClassifier(min_samples_split = 20, max_depth = 15, n_estimators = 500 , random_state = 69)
    randomForest.fit(trainFeatures, trainTarget);
    predictionTest = randomForest.predict(test[list(test.columns[2:])])
    predictionTrain = randomForest.predict(trainFeatures)

    #Answer to question 2.i
    trainConfusion = confusion_matrix(train["target"], predictionTrain)
    testConfusion = confusion_matrix(test["target"], predictionTest)
    print "\nTraining set confusion matrix"
    print trainConfusion
    print "\nTest set confusion matrix"
    print testConfusion
    print "\nTraining set prediction, recal, and f-score"
    print precision_recall_fscore_support(train["target"], predictionTrain, average='binary')[:3]

    #Answer to question 1.iii
    print "\nTest set prediction, recall, and f-score"
    print precision_recall_fscore_support(test["target"], predictionTest, average='binary')[:3]

    #Answer to question 1.iv
    print "\nprecision and recall decrease between the training data and test data."
    print "this is because the decision tree was trained on the training data, which"
    print "resulted in a perfect fit to that specific set. this is an example of why"
    print "you shouldn't train a model on test data\n"

    #Answer to question 1.v
    print "for the train data"
    x = (trainConfusion[1][1])
    y = (x + trainConfusion[1][0])
    print "\tTrue Positive Rate = " + str(float(x)/float(y))
    x = (trainConfusion[0][1])
    y = (x + trainConfusion[0][0])
    print "\tFalse Positive Rate = " + str(float(x)/float(y))
    #Answer to question 1.vi
    print "\nfor the test data"
    x = (testConfusion[1][1])
    y = (x + testConfusion[1][0])
    print "\tTrue Positive Rate = " + str(float(x)/float(y))
    x = (testConfusion[0][1])
    y = (x + testConfusion[0][0])
    print "\tFalse Positive Rate = " + str(float(x)/float(y))

    #Ansrew to question 1.vii
    print "\nagain, testing on the training data results in perfect classification"
    print "of training data, as the model is based off of the test data. The test"
    print "data is different, which exposes some errors in prediction\n"


    return 0


if __name__ == "__main__":
    main()
