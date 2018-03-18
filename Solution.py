# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 02:27:50 2018

@author: Hrithik Singh
"""

from sklearn import tree
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

clf_tree = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...

# 1 K Nearest Neighbors
clf_knn = neighbors.KNeighborsClassifier()

# 2 Support Vector Machines
clf_svm = SVC()

# 3 Gaussian Naive Bayes
clf_nb = GaussianNB()
 
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data

# Training Other Classifiers

clf_knn = clf_knn.fit(X,Y)
clf_tree = clf_tree.fit(X, Y)
clf_svm = clf_svm.fit(X,Y)
clf_nb = clf_nb.fit(X,Y)


prediction1 = clf_tree.predict([[190, 70, 43]])
prediction2 = clf_knn.predict([[190, 70, 43]])
prediction3 = clf_svm.predict([[190, 70, 43]])
prediction4 = clf_nb.predict([[190, 70, 43]])
# CHALLENGE compare their reusults and print the best one!

final = max(prediction4,prediction1,prediction2,prediction3)
#print(prediction1,prediction2,prediction3,prediction4)
if final == prediction1 :
    print("decision tree")
elif final == prediction2 :
    print("K Nearest Neighbors Neighbors")
elif final == prediction3 :
    print("SUpport Vector Machines")
elif final == prediction4 :
    print("Naive Bayes (Gaussian)")
print(final)