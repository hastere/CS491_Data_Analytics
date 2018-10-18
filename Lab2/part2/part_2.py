import csv
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
def main():
    #load data
    train = pd.read_csv('part2Train.csv', header=None)
    test = pd.read_csv('part2Test.csv', header=None)
    scaler = MinMaxScaler()
    target = train[4]
    validation = test[4]
    #scale it
    scaledTrain = scaler.fit_transform(train[list(train.columns[:4])])
    scaledTest = scaler.fit_transform(test[list(train.columns[:4])])
    trainFeatures = pd.DataFrame(scaledTrain)
    testFeatures = pd.DataFrame(scaledTest)

    # testFeatures = test[list(test.columns[:4])]
    # trainFeatures = train[list(train.columns[:4])]
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
    main()
