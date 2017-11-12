import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn import datasets

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Bu çalışma kapsamında "iris" veri seti kullanılacak !

iris = datasets.load_iris()

X = iris.data
y = iris.target

print('Class labels:', np.unique(y))

# Verimizi normalize ediyoruz !

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(X)
X = ss.transform(X)

# Veri setimizi train ve test diye ikiye ayırıyoruz !

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Karar Ağacı algoritması ve elde edilen sonuçlar !

dtree = tree.DecisionTreeClassifier(criterion = 'entropy', random_state=0)
dtree.fit(X_train, y_train)

# generate evaluation metrics
print("Train - Accuracy :", metrics.accuracy_score(y_train, dtree.predict(X_train)))
print("Train - Confusion matrix :",metrics.confusion_matrix(y_train, dtree.predict(X_train)))
print("Train - classification report :", metrics.classification_report(y_train, dtree.predict(X_train)))
print("\n")
print("Test - Accuracy :", metrics.accuracy_score(y_test, dtree.predict(X_test)))
print("Test - Confusion matrix :",metrics.confusion_matrix(y_test, dtree.predict(X_test)))
print("Test - classification report :", metrics.classification_report(y_test, dtree.predict(X_test)))

# Karar ağacının oluşturulması !

import graphviz
dot_data = tree.export_graphviz(dtree, out_file=None,feature_names=iris.feature_names,  
                         class_names=iris.target_names, filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 

dot_data = tree.export_graphviz(dtree, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 



