# Titanic-Survival-Prediction-using-decision-tree-algorithm-from-scratch
#### The RMS Titanic was a luxury steamship and the ship sank on April 15, 1912. In this disaster
more. then 1500 people lost their lives from a total of 2240 people. Out of 126, 59 children aged
under 14 sadly died. However, there was a 53.4% survival rate shown in the titanic facts article
(TitanicFacts, 2020). In this project, my goal is to predict which passengers survived the Titanic
shipwreck using a suitable machine learning model. I have taken the dataset from the Kaggle
website (Kaggle, 2020). Based on the training dataset the survival rate is 38.4%. I have decided to
build a decision tree model to predict survived passengers. I am using the CART (Classification
and Regression Trees) technique to build a Decision Tree model.

#### Data Exploration
Description of features:
• PassengerId: Id of each passanger
• Survived: The status of the passenger survived or not. here 0 = not survived, 1
= survived
• Pclass: Class of passenger based on socio-economic status. here, 1 = Upper, 2
= Middle, 3 = Lower
• Name: Name of passenger
• Sex: Gender of a passenger. values: male and female.
• Age: age of passenger
• SibSp: number of siblings or spouses aboard the Titanic
• Parch: number of parents or children aboard the Titanic
• Ticket: Ticket number
• Fare: fare of passenger
• Cabin: Cabin number
• Embarked: port to board a ship. Here C = Cherbourg, Q = Queenstown, S =
Southampton

#### Data Cleaning is a very important part before we build the model. In the cleaning
part if we have any missing values then it is necessary to handle them. In some
cases, we cannot remove the all missing values, because it causes us to remove
important data. In this case, we can fill values manually or simply by adding zeros.
However, if missing values do not have any dependency concerning other features
then we can directly remove them.

#### Model selection
I have decided to work on the Decision Tree model. The reason is that the decision tree is
faster than other models. For, example, KNN doesn’t support feature interaction whereas
Decision Tree does support. Naïve Bayes is a good model, but if we have categorical data
and if it didn’t observe in training data, the model will assign 0 probability which can lead
to making a wrong prediction. After thinking about all pros and cons of the Decision Tree,
I moved forward to work with the Decision Tree algorithm.

#### Feature Selection
I have chosen Pclass, Gender, Age, SibSp, Parch, Fare, Embarked, Prefix and Survived
features for the model building process.

#### Model building
I will start building the model from the root node. We can select the root node based on
information gain. The higher the information gain the more suitable feature for the root
node. Then we can split the remaining features based on questions we ask. Those nodes
are known as sub-nodes or child nodes of the decision tree. The sub-nodes then become
the two child nodes of the root node. The goal of this process is to separate the labels as
we proceed down. The important fact to build an effective decision tree is based on which
question to ask and when to ask and to do that we have to quantify how many questions
can help us to separate the labels. We can quantify the amount of uncertainty of each node
(feature) using a matrix called Gini impurity and can calculate how much questions reduce
that uncertainty using the Information Gain formula. We use this concept to select the best
question to ask and split the data based on those questions till the end of the questions. The
end nodes are known as a leaf node.
What type of questions can we ask about the data?
Each node takes a list of rows of inputs and to generate the list of questions will iterate
over every value for each feature that appeared in that rows. Each of them becomes the
candidate for a threshold we can use to partition the data and check different possibilities.
How to decide which question asks when?
We can calculate the impurity of the dataset (target feature) and then the impurity of each
feature (dependent features to predict target feature). There are two different ways to
calculate the impurity of features known as entropy index and Gini index. Gini index is
used for CART (Classification and regression tree) and it gives more accurate results as
compare to entropy index.

I have used Entropy gain using below formula:


Then we calculate the Information Gain based on the Reminder and Gini impurity of the
feature. Information gain tells us how important the feature is to predict the results or
labeled data.

Information Gain formula = entropy(parent node) – [average entropy(children node)]

##### Find the best split:
The best split is dependent on information gain. If the information gain is greater, then
it will become a root node. Once we find the root node, we have to divide our dataset
into two categories. It could be true and False, or Yes or No. (Mostly we divide data
into the binary format).

#### Model evaluation
After building the decision tree model and testing the data, I wanted to check the accuracy
score of the built model.
I have only created one decision tree and I got an accuracy score of 80.12. I create
many decision trees and combine them the calculate accuracy score then it will give me a
good accuracy score. The combination of many decision trees is known as a "Random
Forest". Accuracy score is also dependent on train and test dataset split amount. I have
split them into 80% train and 20% test dataset.

#### Confusion Matrix
We can check the performance of classification model using confusion matrix on the test
dataset.
• True Positive (TP): In this case the model predicts 1 and actual value is also 1. It
means passenger survived on Titanic Cruise.
• True Negative (TN): Here, the model predicts 0, and actual value is 0. So, we have
predicted that passenger didn’t survived, which is similar to the actual value.
• False Positive (FP): The model predicts 1, but actual value is 0, which means we
have predicted that the passenger survived on Titanic Cruise, but in actual he/she
didn’t survive.
• False Negative (FN): The model predicts 0, but actual value is 1. It means we have
predicted passenger not survived, but in real he/she survived.

We can see that decision tree model has 98 true positive, 23 False Negative, 11
False Positive and 39 True Negative. So, decision tree model predicted 34 wrong
out of 171 records.

#### Conclusion
Decision tree is one type of supervised machine learning and this model works on
classification and regression problems. Decision Tree model is more useful in case of
classification problem and Titanic dataset is based on classification problem. Decision Tree
is the base of many advanced machine learning models such as Random Forest, AdaBoost,
XGBoost and so on. The steps to learn a decision tree from the scratch can help us to build
a strong base in the machine learning field. The accuracy score of the decision tree is 80.12.
<b> The process of shuffling the data can affect the accuracy score, while re-running the model.</b>
The reason to choose decision tree over other model such as KNN, decision is faster than
KNN model. The future study is to build a Random Forest model from scratch to get a
more accurate prediction.

#### Kaggle Ranking Result
After training a model I have fit the test dataset for Kaggle challenge and found below
result after submitting the predicted csv file. We can see the predicted dataset is 70%
accurate as per the Kaggle challenge and rank score is 0.52631. Kaggle User name is
‘Nilam Shinde1’.
