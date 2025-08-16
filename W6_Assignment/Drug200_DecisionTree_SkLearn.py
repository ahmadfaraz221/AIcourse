# Drug dataset to apply decision tree classification via skit learn
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

col_names = ['Age','Sex','BP','Cholesterol','Na_to_K', 'Drug']
pima = pd.read_csv("W6_Assignment/drug200.csv",header=1, names=col_names)

print(pima.head())

feature_cols = ['Age','Sex','BP','Cholesterol','Na_to_K']
x = pima[feature_cols]
y = pima.Drug

x.loc[:, 'Sex'] = x['BP'].map({'M': 0, 'F': 1})
x.loc[:, 'BP'] = x['BP'].map({'HIGH': 0, 'LOW': 1, 'NORMAL': 2})
x.loc[:, 'Cholesterol'] = x['Cholesterol'].map({'HIGH': 0, 'LOW': 1, 'NORMAL': 2})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=clf.classes_)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DrugV1.png')
Image(graph.create_png())

clf = DecisionTreeClassifier(criterion="entropy",max_depth=3)
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=clf.classes_)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DrugV2.png')
Image(graph.create_png())



