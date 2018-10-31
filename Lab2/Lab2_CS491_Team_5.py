import csv
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn import tree
from sklearn import svm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


#Contributions
#   Stere - Drafted answers to part 1, reviewed answers in part 2
#   Parse - Drafted answers to part 2, reviewed answers in part 1
def main_pt_1():
    print "\t\t\t-Part 1-\n"
    #answers to part one
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
    print "\nTraining set preciction, recall, and f-score"
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
    print "Due to the inbalance in samples the recall value is a better way of evaluating"
    print "performance"

    print "\nNow using RandomForestClassifier as a model"
    #mention that this might be too broad?
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
    #Answer to question 2.ii
    print "\nTraining set precsion, recal, and f-score"
    print precision_recall_fscore_support(train["target"], predictionTrain, average='binary')[:3]

    #Answer to question 2.iii
    print "\nTest set precsion, recall, and f-score"
    print precision_recall_fscore_support(test["target"], predictionTest, average='binary')[:3]

    #Answer to question 2.iv
    print "\nprecision and recall decrease between the training data and test data."
    print "this is because the decision tree was trained on the training data, which"
    print "resulted in a perfect fit to that specific set. this is an example of why"
    print "you shouldn't train a model on test data. The model overfit the training"
    print "data, resulting in poor performance in the test set.\n"

    #Answer to question 2.v
    print "for the train data"
    x = (trainConfusion[1][1])
    y = (x + trainConfusion[1][0])
    print "\tTrue Positive Rate = " + str(float(x)/float(y))
    x = (trainConfusion[0][1])
    y = (x + trainConfusion[0][0])
    print "\tFalse Positive Rate = " + str(float(x)/float(y))
    #Answer to question 2.vi
    print "\nfor the test data"
    x = (testConfusion[1][1])
    y = (x + testConfusion[1][0])
    print "\tTrue Positive Rate = " + str(float(x)/float(y))
    x = (testConfusion[0][1])
    y = (x + testConfusion[0][0])
    print "\tFalse Positive Rate = " + str(float(x)/float(y))

    #Ansrew to question 2.vii
    print "\nagain, testing on the training data results in perfect classification"
    print "of training data, as the model is based off of the test data. The test"
    print "data is different, which exposes some errors in prediction\n"


    return 0

def main_pt_2():
    print "\t\t\t-Part 2-\n"
    #answers to part two
    #load data
    train = pd.read_csv('part2Train.csv', header=None)
    test = pd.read_csv('part2Test.csv', header=None)

    #scale it
    scaler = MinMaxScaler()
    scaledTrain = scaler.fit_transform(train[list(train.columns[:4])])
    scaledTest = scaler.fit_transform(test[list(train.columns[:4])])

    #assign features and targets
    trainFeatures = pd.DataFrame(scaledTrain)
    testFeatures = pd.DataFrame(scaledTest)
    target = train[4]
    validation = test[4]
    #decision tree
    dataTree = tree.DecisionTreeClassifier(random_state=69)
    dataTree = dataTree.fit(trainFeatures, target)
    dataTreePrediction = dataTree.predict(testFeatures)
    dataTreeConfusion = confusion_matrix(validation, dataTreePrediction)
    dataTreeMetrics = precision_recall_fscore_support(validation, dataTreePrediction, average='binary')[:3]
    #svm
    svmModel = svm.LinearSVC(random_state=69)
    svmModel = svmModel.fit(trainFeatures, target)
    svmPrediction = svmModel.predict(testFeatures)
    svmConfusion = confusion_matrix(validation, svmPrediction)
    svmMetrics = precision_recall_fscore_support(validation, svmPrediction, average='binary')[:3]
    #logistic regression
    logReg = LogisticRegression(random_state=69, solver='lbfgs', multi_class='multinomial')
    logReg = logReg.fit(trainFeatures, target)
    logRegPrediction = logReg.predict(testFeatures)
    logRegConfusion = confusion_matrix(validation, logRegPrediction)
    logRegMetrics = precision_recall_fscore_support(validation, logRegPrediction, average='binary')[:3]
    #artificial neural net
    annModel = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=69)
    annModel = annModel.fit(trainFeatures, target)
    annPrediction = annModel.predict(testFeatures)
    annConfusion = confusion_matrix(validation, annPrediction)
    annMetrics = precision_recall_fscore_support(validation, annPrediction, average='binary')[:3]
    print "\n"
    #Answer to q 3.i
    print "Data Tree confusion matrix"
    print dataTreeConfusion
    print "Data Tree metrics"
    print dataTreeMetrics
    print "\n"
    #Answer to q 3.ii
    print "SVM confusion matrix"
    print svmConfusion
    print "SVM metrics"
    print svmMetrics
    print "\n"
    #Answer to q 3.iii
    print "Logistic Regression confusion matrix"
    print logRegConfusion
    print "Logistic Regression metrics"
    print logRegMetrics
    print "\n"
    #Answer to q 3.iv
    print "Artifical Neural Network confusion matrix"
    print annConfusion
    print "Artifical Neural Network metrics"
    print annMetrics
    print "\n"
    return 0

if __name__ == "__main__":
    main_pt_1()
    main_pt_2()
