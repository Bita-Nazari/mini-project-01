# Fraud Detection

## Project Goal

The goal of this project is to detect fraudulent transactions.

We have a classification problem.

The dataset source is https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

In this project, we will preprocess the data, train different classification models, evaluate their performance, and compare the results.

The models used in this project are:
- Logistic Regression
- KNN
- Decision Tree

I guess logistic regression will perform better than the others bc knn model is not a good choice for this dataset we have few fraud samples and i think its not gonna work well here , knn may give us good accuracy but its not gonna give us good recall
and bc our dataset is really big knn will be a very lazy learner in this situation and its gonna calculate distance to training samples and decision tree may not work well either bc if we set a big max depth its going to overfit and if we set a small max depth its gonna fet underfit but with a normal max depth it may work good

Based on to the problrm recall is more important than the others bc its really important for us to detect fraud transactiona as many as possible we can

our data is really imbalanced so most of our samples are legitimate  and bc of this imbalanced data accurecy will be high but it doesnt mean model works well. it gets high bc most of our samples are not fraud

I expect feature scaling to significantly affect KNN performance bc if we dont scale features some features value might be very big and it affect
model prediction especially in knn 

if we set large max depth its gonna overfit bc it makes model complicated and its gonna work well only in train sets


after evaluation models
all of the models achieved high accuracy , and this is because of our imbalanced data
most of the samples are not fraud and models detect most of the samples correctly and achieve high accuracy but it may not work well at detecting fraud samples
despite what i said before testing the models ,logistic regression didnt work well in this case,and logistic regression couldnt fit to the data ,base on this logistic regression its out of the list even with the high accuracy bc it didnt work well in the other metrics ,based on the results Decision tree and knn model both of them work well and base on the cross validation result knn may work better , knn had better precision and better f1 score but decision tree had a better recall , i chose decision tree for this case bc decision tree worked better at the thing that i want --> predicting more fraud
the base of this problem is that we could detect fraud transaction, so in this case recall is the most important metric for us,and its important for us to find more and more fraud transactions even ,if predict some of them wrong ,based on this opinion decision tree its the better choice

Decision Tree has a higher Recall but lower Precision than KNN. it detects more actual fraud transactions, but it also classifies more legitimate transactions as fraud.

with more experiments Random Forest worked better than the other

For a fraud detection system, the threshold should be selected according to the main objective of the project.

When the threshold decreases, the model becomes more likely to classify a transaction as fraudulent. Therefore, Recall increases and more fraudulent transactions are detected.

However, decreasing the threshold also increases the number of transactions that are incorrectly classified as fraudulent, which causes Precision to decrease.

SIts true recall is very important but we have to have balanced model too , good realation between recall and percision i think its better to use 0.5 treshold for random forest model.