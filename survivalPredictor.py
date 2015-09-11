# script file for predicting titanic survival rates

import pandas
# import csv
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# reads csv files through csv module
'''
f1 = open("train.csv")
f2 = open("genderclassmodel.csv")
f3 = open("gendermodel.csv")
try:
    trainingDataReader = csv.reader(f1) #reads training data
    genderClassReader = csv.reader(f2)
    genderModelReader = csv.reader(f3)
    for row in trainingDataReader:
        print(row)

finally:
    f1.close()
    f2.close()
    f3.close()
'''


# reads csv files through pandas module
titanic_training = pandas.read_csv("train.csv")
titanic_test = pandas.read_csv("test.csv")
# print(titanic_training.as_matrix())


# fill missing age values with the median age value
median_age = titanic_training["Age"].median()
titanic_training["Age"] = titanic_training["Age"].fillna(median_age)

# replace male with 0, replace female with 1
titanic_training.loc[titanic_training["Sex"] == "male", "Sex"] = 0
titanic_training.loc[titanic_training["Sex"] == "female", "Sex"] = 1

# replace embarked values
titanic_training["Embarked"] = titanic_training["Embarked"].fillna("S")
titanic_training.loc[titanic_training["Embarked"] == "S", "Embarked"] = 0
titanic_training.loc[titanic_training["Embarked"] == "C", "Embarked"] = 1
titanic_training.loc[titanic_training["Embarked"] == "Q", "Embarked"] = 2

# convert test data to number values
'''
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
'''

# predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
predictors = ["Sex", "Age", "Embarked"]
predict = ["Survived"]

'''
# configure Naive Bayes classifier
data = datasets.load_iris()
print(data.target)
# print(data.target)
'''

# build naive bayes classifier
classifier = GaussianNB()
classifier.fit(titanic_training.as_matrix(predictors), titanic_training.as_matrix(predict)) # x = features, y = labels
predictions = classifier.predict(titanic_test.as_matrix(predictors))

'''
# generate kaggle submission
submission = pandas.DataFrame({
    "PassengerId" : titanic_test["PassengerID"],
    "Survived" : predictions
})

submission.to_csv("kaggle_titanic_submission", index=False)
'''