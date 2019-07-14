# Importing The Dataset

import pandas as pd


def funprint(yp):
    for i in yp:
        if i == 0:
            print("Iris-setosa")
        elif i == 1:
            print("Iris-versicolor")
        elif i == 2:
            print("Iris-virginica")


# Importing the dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, [0, 1, 2, 3]].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Fitting Random Forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
funprint(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))