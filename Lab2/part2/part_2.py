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

def main():
    #load data
    train = pd.read_csv('part2Train.csv', header=None)
    test = pd.read_csv('part2Test.csv', header=None)
    target = train[4]
    features = list(train.columns[:4])
    catData = train[features]
    #decision tree
    dataTree = tree.DecisionTreeClassifier()
    dataTree = dataTree.fit(catData, target)
    dataTreePrediction = dataTree.predict(test[list(test.columns[:4])])
    dataTreeConfusion = confusion_matrix(test[4], dataTreePrediction)
    dataTreeMetrics = precision_recall_fscore_support(test[4], dataTreePrediction, average='binary')[:3]
    #svm
    svmModel = svm.LinearSVC()
    svmModel = svmModel.fit(catData, target)
    svmPrediction = svmModel.predict(test[list(test.columns[:4])])
    svmConfusion = confusion_matrix(test[4], svmPrediction)
    svmMetrics = precision_recall_fscore_support(test[4], svmPrediction, average='binary')[:3]
    #logistic regression
    logReg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    logReg = logReg.fit(catData, target)
    logRegPrediction = logReg.predict(test[list(test.columns[:4])])
    logRegConfusion = confusion_matrix(test[4], logRegPrediction)
    logRegMetrics = precision_recall_fscore_support(test[4], logRegPrediction, average='binary')[:3]
    #artificial neural net
    annModel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    annModel = annModel.fit(catData, target)
    annPrediction = annModel.predict(test[list(test.columns[:4])])
    annConfusion = confusion_matrix(test[4], annPrediction)
    annMetrics = precision_recall_fscore_support(test[4], annPrediction, average='binary')[:3]
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
