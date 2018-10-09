import csv
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier



def main():
    #load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    target = train["target"]
    features = list(train.columns[3:])
    catData = train[features]
    dataTree = tree.DecisionTreeClassifier()
    dataTree = dataTree.fit(catData, target)
    predictionTest = dataTree.predict(test[list(test.columns[3:])])
    predictionTrain = dataTree.predict(catData)

    #Answer to question 1.i
    trainConfusion = confusion_matrix(train["target"], predictionTrain)
    testConfusion = confusion_matrix(test["target"], predictionTest)
    print "Training set confusion matrix"
    print trainConfusion
    print "Test set confusion matrix"
    print testConfusion
    #Answer to question 1.ii

    print "Training set prediction, recal, and f-score"
    print precision_recall_fscore_support(train["target"], predictionTrain, average='binary')[:3]

    #Answer to question 1.iii
    print "Test set prediction, recall, and f-score"
    print precision_recall_fscore_support(test["target"], predictionTest, average='binary')[:3]

    #Answer to question 1.iv
    print "precision and recall decrease between the training data and test data."
    print "this is because the decision tree was trained on the training data, which"
    print "resulted in a perfect fit to that specific set. this is an example of why"
    print "you shouldn't train a model on test data"

    #Answer to question 1.v
    print "for the train data"
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
    print "again, testing on the training data results in perfect classification"
    print "of training data, as the model is based off of the test data. The test"
    print "data is different, which exposes some errors in prediction"

    #Answer to question 1.viii
    print "Due to variance in samples the recall value is a better way of evaluating"
    print "performance"

    print "Now using RandomForestClassifier as a model"
    randomForest = RandomForestClassifier(n_estimators = 100, random_state = 42)
    randomForest.fit(catData, target);
    predictionTest = randomForest.predict(test[list(test.columns[3:])])
    predictionTrain = randomForest.predict(catData)

    #Answer to question 2.i
    trainConfusion = confusion_matrix(train["target"], predictionTrain)
    testConfusion = confusion_matrix(test["target"], predictionTest)
    print "Training set confusion matrix"
    print trainConfusion
    print "Test set confusion matrix"
    print testConfusion
    print "Training set prediction, recal, and f-score"
    print precision_recall_fscore_support(train["target"], predictionTrain, average='binary')[:3]

    #Answer to question 1.iii
    print "Test set prediction, recall, and f-score"
    print precision_recall_fscore_support(test["target"], predictionTest, average='binary')[:3]

    #Answer to question 1.iv
    print "precision and recall decrease between the training data and test data."
    print "this is because the decision tree was trained on the training data, which"
    print "resulted in a perfect fit to that specific set. this is an example of why"
    print "you shouldn't train a model on test data"

    #Answer to question 1.v
    print "for the train data"
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
    print "again, testing on the training data results in perfect classification"
    print "of training data, as the model is based off of the test data. The test"
    print "data is different, which exposes some errors in prediction"
        



    return 0


if __name__ == "__main__":
    main()
