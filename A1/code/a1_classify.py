from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from scipy import stats
import numpy as np
import argparse
import sys
import os

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    total_TP = np.trace(C)
    total_classified = np.sum(C)
    return total_TP/total_classified

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    TP_array = np.diag(C)
    TP_FN_array = np.sum(C, axis=0)
    return TP_array/TP_FN_array

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    TP_array = np.diag(C)
    TP_FP_array = np.sum(C, axis=1)
    return TP_array/TP_FP_array
    

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    data = np.load(filename)["arr_0"]
    col_split = np.hsplit(data, [173])
    X = col_split[0]
    y = col_split[1].T[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 5 classifiers
    linear_SVC = LinearSVC()
    rbf_SVC = SVC(kernel="rbf", gamma=2)
    random_forest_classifier = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
    mlp_classifier = MLPClassifier(alpha=0.05)
    ada_boost_classifier = AdaBoostClassifier()

    # fit classifiers
    linear_SVC.fit(X_train, y_train)
    rbf_SVC.fit(X_train, y_train)
    random_forest_classifier.fit(X_train, y_train)
    mlp_classifier.fit(X_train, y_train)
    ada_boost_classifier.fit(X_train, y_train)

    # predict data
    one = linear_SVC.predict(X_test)
    two = rbf_SVC.predict(X_test)
    three = random_forest_classifier.predict(X_test)
    four = mlp_classifier.predict(X_test)
    five = ada_boost_classifier.predict(X_test)
    y_pred = [one, two, three, four, five]

    # get confusion matrix and accuracy
    confused = []
    accs = []
    for i in range(5):  
        C = confusion_matrix(y_test, y_pred[i])
        accs.append(accuracy(C))
        confused.append(C)

    # write to csv
    with open('a1_3.1.csv', 'w') as f:
        for i in range(1,6):
            C = confused[i-1]
            a = accuracy(C)
            r = recall(C)
            p = precision(C)
            f.write(str(i)+','+str(a)+','+str(r)[1:-1]+','+str(p)[1:-1]+','+str(list(C.flatten()))[1:-1]+'\n')
    
    iBest = np.argmax(accs)+1 

    return (X_train, X_test, y_train, y_test,iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    # 5 classifiers
    classifier_set = {1: LinearSVC(), 2: SVC(kernel="rbf", gamma=2), 3: RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0), 4: MLPClassifier(alpha=0.05),5: AdaBoostClassifier()}
    sample_size = [1000, 5000, 10000, 15000, 20000]
    y_pred = []
    for x in sample_size:
        classy = classifier_set[iBest]
		
		# sample data 
        idxs = np.random.choice(X_train.shape[0], x, replace=False)
        if x == 1000:
            X_1k = X_train[idxs, :]
            y_1k = y_train[idxs]
        X = X_train[idxs, :]
        y = y_train[idxs]
		#check x and y shape
        classy.fit(X, y)
        y_pred.append(classy.predict(X_test))
	
	# get confusion matrix and accuracy
    accs = []
    for i in range(5):  
        C = confusion_matrix(y_test, y_pred[i])
        accs.append(accuracy(C))
    
    # write to csv
    with open('a1_3.2.csv', 'w') as f:
        f.write(str(accs)[1:-1]+'\n') #first line of accuracies
        f.write("We can definitely see a trend that accuracy increases as number of elements in a training set increases, however slightly goes lower on last one. The reason why we achieve better accuracy with more data is because with more data we are able to achieve better generalization. However we can also observe for the largest data set that there is a diminishing value added at some point, so more data doesn't always mean better results in accuracy. ")
        
    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    classifier_set = {1: LinearSVC(), 2: SVC(kernel="rbf", gamma=2), 3: RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0), 4: MLPClassifier(alpha=0.05),5: AdaBoostClassifier()}
    ks = [5, 10, 20, 30, 40, 50]
    # write to csv
    with open('a1_3.3.csv', 'w') as f:
        for k in ks:
            selector1k = SelectKBest(f_classif, k)
			# fit 1k
            X1k_new = selector1k.fit_transform(X_1k, y_1k)
            feat51k_id = selector1k.get_support(True)
			# fit 32k
            selector = SelectKBest(f_classif, k)
            X_new = selector.fit_transform(X_train, y_train)

            pp = selector.pvalues_
			#find best k features
            feat_id = selector.get_support(True)
            if k == 5:
                X1k_f5 = X1k_new
                X_f5 = X_new
                feat5_id = feat_id
                feat51k_id = feat51k_id
            ppk = []
            for i in feat_id:
                ppk.append(pp[i])
                
            #line 1-6	
            f.write(str(k)+','+str(ppk)[1:-1]+'\n')
	
        a = []
        # find accuracy for k=5 and data 1k
        classy1k = classifier_set[iBest]
        classy1k.fit(X1k_f5, y_1k)
        X_test_new1k = np.take(X_test,feat51k_id,axis=1)
        y_pred = classy1k.predict(X_test_new1k)
        C = confusion_matrix(y_test, y_pred)
        a.append(accuracy(C))

        # find accuracy for k=5 and data 32k
        classy = classifier_set[iBest]
        classy.fit(X_f5, y_train)
        X_test_new = np.take(X_test,feat5_id,axis=1)
        y_pred = classy.predict(X_test_new)
        C = confusion_matrix(y_test, y_pred)
        a.append(accuracy(C))

        #line 7
        f.write(str(a)[1:-1]+'\n')
        #line 8-10


def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    data = np.load(filename)["arr_0"]
    col_split = np.hsplit(data, [173])
    X = col_split[0]
    y = col_split[1].T[0]
    
    # 5 classifiers
    linear_SVC = LinearSVC()
    rbf_SVC = SVC(kernel="rbf", gamma=2)
    random_forest_classifier = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
    mlp_classifier = MLPClassifier(alpha=0.05)
    ada_boost_classifier = AdaBoostClassifier()
    
    csv = np.zeros([5,5])
    i=0
    kf = KFold(n_splits=5, shuffle=True)
    with open('a1_3.4.csv', 'w') as f:
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # fit classifiers
            linear_SVC.fit(X_train, y_train)
            rbf_SVC.fit(X_train, y_train)
            random_forest_classifier.fit(X_train, y_train)
            mlp_classifier.fit(X_train, y_train)
            ada_boost_classifier.fit(X_train, y_train)

            # predict data
            one = linear_SVC.predict(X_test)
            two = rbf_SVC.predict(X_test)
            three = random_forest_classifier.predict(X_test)
            four = mlp_classifier.predict(X_test)
            five = ada_boost_classifier.predict(X_test)
            y_pred = [one, two, three, four, five]

            # get confusion matrix and accuracy
            accs = []
            for i in range(5):  
                C = confusion_matrix(y_test, y_pred[i])
                accs.append(accuracy(C))

            #report accuracies for all classifiers on each fold
            csv[i] = accs
            f.write(str(accs)[1:-1]+'\n')
	
    # report p values
    pvals = []
    for i in range(1,5):
        S = stats.ttest_rel(csv[:][0], csv[:][i])	
        pvals.append(S)
    f.write(str(pvals)[1:-1])
	   
	   
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
    # 3.1 Classifiers
    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    # 3.2 Amount of training data
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test,iBest)
    
    # 3.3 Feature analysis
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    # 3.4 Cross-validation
    class34(args.input, iBest)