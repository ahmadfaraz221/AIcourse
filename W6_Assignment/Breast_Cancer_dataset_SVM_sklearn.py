# Breast Cancer dataset SVM via sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_link = "W6_Assignment/breast-cancer.csv"
col_names = ["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean",
             "concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se",
             "perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se",
             "fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
             "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

cancerdata = pd.read_csv(data_link, names=col_names, sep=",", header=None)
cancerdata.head()

print(cancerdata['diagnosis'].unique())
print(cancerdata.shape)

print ( " Exploring the Dataset:  cancerdata['diagnosis'].value_counts()) \n " , cancerdata['diagnosis'].value_counts())
print ( " Exploring the Dataset:  cancerdata['diagnosis'].value_counts()) \n " , cancerdata['diagnosis'].value_counts(normalize=True) )


cancerdata['diagnosis'] = cancerdata['diagnosis'].map({'M': 1, 'B': 0})
cancerdata['diagnosis'].plot.hist()
plt.show()

print("cancerdata.describe().T   :    \n" , cancerdata.describe().T )
for col in cancerdata.columns:
    if pd.api.types.is_numeric_dtype(cancerdata[col]):  # check if numeric
        cancerdata[col].plot.hist(title=col, bins=30)
        plt.xlabel(col)
        plt.show()


y = cancerdata['diagnosis']
X = cancerdata.drop('diagnosis', axis=1)

from sklearn.model_selection import train_test_split

SEED = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = SEED)

xtrain_samples = X_train.shape[0]
xtest_samples = X_test.shape[0]

print(f'There are {xtrain_samples} samples for training and {xtest_samples} samples for testing.')
from sklearn.svm import SVC

svc = SVC(kernel='linear', random_state=42)  
svc.fit(X_train, y_train)

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(y_test,y_pred)
gg=sns.heatmap(cm, annot=True, fmt='d').set_title('Confusion matrix of linear SVM') 
print(classification_report(y_test,y_pred))