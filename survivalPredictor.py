# script file for predicting titanic survival rates

import pandas
import numpy
from ggplot import *

from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.linear_model import LinearRegression, LogisticRegression  # Linear Regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import cross_validation # Splitting the data set

from sklearn.feature_selection import SelectKBest, f_classif

# reads csv files through pandas module
titanic_training = pandas.read_csv("train.csv") # dataframe object
titanic_test = pandas.read_csv("test.csv")

accuracyPercentages = {}

# fill missing age values with the median age value
median_age = titanic_training["Age"].median()
titanic_training["Age"] = titanic_training["Age"].fillna(median_age)
titanic_training["Fare"] = titanic_training["Fare"].fillna(titanic_training["Fare"].median())

# replace male with 0, replace female with 1
titanic_training.loc[titanic_training["Sex"] == "male", "Sex"] = 0
titanic_training.loc[titanic_training["Sex"] == "female", "Sex"] = 1

# replace embarked values
titanic_training["Embarked"] = titanic_training["Embarked"].fillna("S")
titanic_training.loc[titanic_training["Embarked"] == "S", "Embarked"] = 0
titanic_training.loc[titanic_training["Embarked"] == "C", "Embarked"] = 1
titanic_training.loc[titanic_training["Embarked"] == "Q", "Embarked"] = 2

# print(titanic_training.describe());
print("Training Data: \n")


headers = list(titanic_training.columns.values)
headers.remove('PassengerId')
headers.remove('Survived')
headers.remove('Cabin')
headers.remove('Name')
headers.remove('Ticket')
print(str(headers))


predictors = []

selector = SelectKBest(f_classif, k = 'all')
selector.fit(titanic_training[headers], titanic_training["Survived"])

for value in zip(headers, selector.pvalues_):
    if value[1] <= 0.05:
        category = value[0]
        predictors.append(category)
        # print(category + ": " + str(value[1]))


kf = cross_validation.KFold(titanic_training.shape[0], n_folds = 3, random_state= 1)
predictions = []

# Linear Regression
alg = LinearRegression()

for train, test in kf:
    train_predictors = (titanic_training[predictors].iloc[train,:])
    train_target = (titanic_training["Survived"].iloc[train])
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic_training[predictors].iloc[test, :])
    predictions.append(test_predictions)

predictions = numpy.concatenate(predictions, axis = 0)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0

accuracy = sum(predictions[predictions == titanic_training["Survived"]]) / len(predictions)
print("Linear Regression: " + str(accuracy))
accuracyPercentages["Linear Regression"] = accuracy



# Logistic Regression
alg2 = LogisticRegression(random_state = 1)
scores = cross_validation.cross_val_score(alg2, titanic_training[predictors], titanic_training["Survived"], cv = 3)
print("Logistic Regression: " + str(scores.mean()))
accuracyPercentages["Logistic Regression"] = scores.mean()


# Naive Bayes
classifier = GaussianNB()
# classifier.fit(titanic_training.as_matrix(predictors), titanic_training.as_matrix(predict)) # x = features, y = labels
# predictions = classifier.predict(titanic_test.as_matrix(predictors))
bayesScores = cross_validation.cross_val_score(classifier, titanic_training[predictors], titanic_training["Survived"], cv = 3)
print("Naive Bayes: " + str(bayesScores.mean()))
accuracyPercentages["Naive Bayes"] = bayesScores.mean()

# Random Forests
alg3 = RandomForestClassifier(random_state = 1, n_estimators = 150, min_samples_split = 2, min_samples_leaf = 1)
randomForestScores = cross_validation.cross_val_score(alg3, titanic_training[predictors],
                                                      titanic_training["Survived"], cv = 3)
print("Random Forests: " + str(randomForestScores.mean()))
accuracyPercentages["Random Forests"] = randomForestScores.mean()


# Gradient Boosting Classifier
alg4 = GradientBoostingClassifier(random_state = 1, n_estimators = 50, max_depth = 3)
gradientBoostScores = cross_validation.cross_val_score(alg4, titanic_training[predictors], titanic_training["Survived"], cv = 3)
print("Gradient Boosting: " + str(gradientBoostScores.mean()))
accuracyPercentages["Gradient Boosting"] = gradientBoostScores.mean()

def draw_plot(accuracyPercentages):
    x_labels = list(accuracyPercentages.keys())
    x_values = range(len(x_labels))
    y_values = []
    for value in x_labels:
        y_values.append(accuracyPercentages[value])
    dataFrame = pandas.DataFrame({"x" : x_values, "y" : y_values})
    print(ggplot(aes(x = "x", y = "y", weight='y'), data = dataFrame) +
          geom_bar() +
          scale_x_continuous(breaks = x_values, labels = x_labels) +
          xlab('classifier') +
          ylab('accuracy score'))

draw_plot(accuracyPercentages)


'''
# apply changes to test data
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
'''


'''
# generate kaggle submission
submission = pandas.DataFrame({
    "PassengerId" : titanic_test["PassengerID"],
    "Survived" : predictions
})

submission.to_csv("kaggle_titanic_submission", index=False)
'''