# Fraudulent Transaction Detection

## 1. Introduction

### 1.1 Project Goal

**Project Goal: Fraudulent Transaction Detection**

In this project, our goal is to build a model that can distinguish fraudulent transactions from non-fraudulent transactions with the highest possible accuracy.

### 1.2 Challenge: Class Imbalance

**Challenge: Class Imbalance**

The number of fraudulent transaction samples is very small, which makes it difficult for the model to correctly identify fraudulent transactions.

Due to the class imbalance, the model can achieve high accuracy; however, because there are very few fraudulent transaction samples, it becomes difficult for the model to correctly identify fraudulent transactions.

---

## 2. Dataset

### 2.1 Introduction to the Dataset

**Introduction to the Dataset**

This dataset is a fraud detection dataset.

It contains 1,081 duplicate samples and no missing values. The dataset consists of 30 features and 284,807 samples.

Among these samples, 492 are fraudulent transactions and 284,315 are non-fraudulent transactions.

Class 0 represents non-fraudulent transactions, while Class 1 represents fraudulent transactions.

---

## 3. Exploratory Data Analysis (EDA)

The first step in examining the data is to draw a class distribution diagram.

As can be seen from the diagram, the class data is highly imbalanced, and the number of fraudulent transaction samples is very low compared to non-fraudulent samples.

### 3.1 Class Distribution

**Class Distribution Diagram**

<!-- Add class distribution image here -->

![Class Distribution](../images/Data_Distribution.png)

99.82% of the samples are non-fraudulent samples, and only about 0.172% of the samples are fraudulent transaction samples, which indicates that the data is highly imbalanced.

### 3.2 Missing Values

In the next step, the observations were checked to see if there was any missing or null data in the dataset, which was found to be absent.

### 3.3 Duplicate Samples

In the next step, the number of duplicate samples was checked. In this dataset, there were 1,081 duplicate samples.

### 3.4 Outlier Analysis

To examine the outlier data, since the values V1 to V28 are features whose type and nature are unknown, and removing the outlier data of these features could disrupt the learning process, none of them were removed.

However, the outlier values of the Amount attribute were examined. There were 31,685 outlier samples, of which only 87 were fraudulent transactions, indicating that outlier data in the Amount attribute does not have a significant impact on the fraudulent nature of transactions.

### 3.5 Feature Selection

After training the Decision Tree model, the feature importance scores were examined to determine which features contributed more to the model's predictions.

Features with higher importance scores were selected and used in the subsequent modeling steps. This allowed the model to focus on the features that had a greater contribution to its predictions while reducing the number of features used for training.

<!-- Add Amount outlier visualization here -->
# 4. Data Preprocessing

## 4.1 Duplicate Removal

In the first stage of data preprocessing, duplicate samples were removed. Removing duplicate samples was necessary because if identical observations were present in both the training and testing sets, they could lead to data leakage and result in an overly optimistic evaluation of the model.

## 4.2 Train-Test Split

After removing the duplicate samples, the dataset was divided into two sets: a training set and a testing set.

Due to the class imbalance in the dataset, stratified splitting was used to preserve the class distribution between the training and testing sets.

The `test_size` was set to 0.2, meaning that 80% of the data was used for training and 20% was reserved for testing. This allowed the model to use a larger portion of the available data for training while keeping a separate test set for final evaluation.

The `random_state` was set to 42 to ensure reproducibility.

```python
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

## 4.3 Feature Selection

After splitting the dataset, feature importance was evaluated using the training data and the Decision Tree model.

The features with higher feature importance scores were selected as input features for the subsequent modeling steps.

The selected features were:

```python
 ['V14', 'V17', 'V10', 'V12', 'V4', 'V28', 'V26', 'V27', 'Time', 'V7', 'V20', 'V8', 'V9', 'V25', 'V1', 'V6', 'V22', 'V18', 'V21', 'V2']
```

These features were selected because they had higher importance scores in the Decision Tree model and therefore contributed more to its predictions.

## 4.4 Feature Scaling

StandardScaler was used to standardize the scale of the features. Scaling is particularly important for models such as KNN and Logistic Regression, where differences in feature scales can affect the model's performance.

The scaler was fitted only on the training data and then used to transform both the training and testing data. Fitting the scaler on the test data could lead to data leakage because information from the test set would be used during the preprocessing stage.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

In this process, `fit_transform()` was applied only to the training data, while the test data was transformed using the parameters learned from the training data.

---

## 4.5 Handling Imbalanced Data

One of the main challenges of this project is the severe imbalance between fraudulent and non-fraudulent transactions.

Due to this class imbalance, a model can achieve a high accuracy by predicting most transactions as non-fraudulent. However, high accuracy does not necessarily indicate good model performance in this case, because the main objective is to correctly identify fraudulent transactions.

In this project, accuracy is therefore not considered an appropriate metric for evaluating the model's performance. Since the model is intended to detect fraudulent transactions, it is important for the model to identify as many fraudulent transactions as possible.

For this reason, Recall is considered one of the most important evaluation metrics in this project. A higher Recall means that the model is able to identify a larger proportion of the actual fraudulent transactions.

The goal is to maximize Recall as much as possible while maintaining a reasonable balance between Recall and Precision and avoiding an excessive increase in false positive predictions.

To address the class imbalance and improve the model's ability to detect the minority class, different sampling techniques were investigated. These techniques included oversampling, undersampling, and combined sampling methods.

The sampling techniques used in this project were:

### Oversampling

* SMOTE



# 5 First Experiment: Baseline Experiment

At this stage, we evaluate the performance of the models on the data without applying any special settings or optimization.

At this stage, we examine the evaluation metrics, including Recall, Precision, F1 score, and the confusion matrix of each model.

## Initial Results

The initial results obtained from a specific split of the data for each model are as follows:

### 5.1 Accuracy

| Model               | Accuracy Train | Accuracy Test |
| ------------------- | -------------: | ------------: |
| Logistic Regression |       0.999207 |      0.999119 |
| KNN Model           |       0.999599 |      0.999418 |
| Decision Tree       |       0.999612 |      0.999330 |
| Random Forest       |       0.999546 |      0.999436 |

### 5.2 Precision

| Model               | Precision Train | Precision Test |
| ------------------- | --------------: | -------------: |
| Logistic Regression |        0.877863 |       0.846154 |
| KNN Model           |        0.949843 |       0.955882 |
| Decision Tree       |        0.942073 |       0.880000 |
| Random Forest       |        0.943662 |       0.943662 |

### 5.3 Recall

| Model               | Recall Train | Recall Test |
| ------------------- | -----------: | ----------: |
| Logistic Regression |     0.608466 |    0.578947 |
| KNN Model           |     0.801587 |    0.684211 |
| Decision Tree       |     0.817460 |    0.694737 |
| Random Forest       |     0.761905 |    0.705263 |

### 5.4 F1 Score

| Model               | F1 Train |  F1 Test |
| ------------------- | -------: | -------: |
| Logistic Regression | 0.718750 | 0.687500 |
| KNN Model           | 0.869440 | 0.797546 |
| Decision Tree       | 0.875354 | 0.776471 |
| Random Forest       | 0.848306 | 0.807229 |

###  5.5 Confusion Matrix Results

### Logistic Regression Model

![Logistic Regression Confusion Matrix](../images/BaseLineExperiment_Images/LG_WithoutFeatureScaling_CM.png)

### KNN Model

![KNN Confusion Matrix](../images/BaseLineExperiment_Images/KNN_WithoutFeatureScaling_CM.png)

### Decision Tree Model

![Decision Tree Confusion Matrix](../images/BaseLineExperiment_Images/DT_WithoutFeatureScaling_CM.png)

### Random Forest Model

![Random Forest Confusion Matrix](../images/BaseLineExperiment_Images/RF_WithoutFeatureScaling_CM.png)

However, since these results were obtained from a specific split of the data, they may not be completely reliable. Therefore, cross-validation is necessary to obtain a more reliable evaluation of the models' performance.

### 5.6 Cross-Validation Results

The results of cross-validation are as follows:

| Model | Mean Accuracy | Mean Recall | Mean Precision | Mean F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.999129 | 0.589474 | 0.869811 | 0.680979 |
| KNN Model | 0.999285 | 0.729115 | 0.848228 | 0.775246 |
| Decision Tree | 0.800663 | 0.779910 | 0.713083 | 0.640749 |
| Random Forest | 0.999337 | 0.725062 | 0.889420 | 0.783736 | 

Based on the cross-validation results, all models performed worse than in the initial single-split evaluation. Among the evaluated models, the Random Forest model achieved the best overall performance in cross-validation.

Therefore, the Random Forest model was selected for further experiments.


## 6. Experiment 02: Class Weight

### 6.1 Experiment Objective

We observed the results without feature selection.

Since the model is highly imbalanced, one of the methods that can be applied is `class_weight='balanced'`.

In this method, the model assigns more weight to the class that has fewer examples, which in this case is class 1 (Fraud).

Therefore, it is worth examining the performance of the model in both cases.

The Random Forest model was selected for further experiments because it achieved the highest F1 score in cross-validation.

### 6.2 Comparison of Normal and Weighted Methods

In the first stage, I compared the model in two ways:

1. Normal
2. Weighted

The purpose of this comparison was to examine the difference in model performance between the normal mode and the mode using:

```python
class_weight='balanced'
```

In the second method, when `class_weight='balanced'` is applied, the model gives more importance to the samples belonging to the minority class, which in this case is class 1 (Fraud).

The advantage of this approach is that the Recall can increase, allowing more fraudulent transactions to be detected.

### 6.3 Results

The results obtained in both cases are as follows:

**Method 1: Normal**

![Not Balanced Data](../images/ClassWeight_Images/weight_notbalanced.png)

**Method 2: Weighted**

![Balanced Data](../images/ClassWeight_Images/weight_balanced.png)

### 6.4 Analysis of the Results

As is clear from the results obtained, in the second method, both TP and FP have increased. This indicates that with weighting, Recall increases while Precision decreases sharply.

However, without weighting, although the Recall value is relatively lower and there are fewer TP values, there are also fewer FP values. Therefore, the Precision and Recall values are closer to each other, resulting in a more balanced model performance.

### 6.5 Conclusion

Based on the conclusion drawn from this section, my choice here is the method that provides a more balanced Precision and Recall.

It is true that Recall is our most important criterion in this project, but this improvement in Recall should not come at the cost of a very low Precision and an unbalanced model performance.

Therefore, my choice is the first method, the method without weighting.


## 7. Experiment 03: Feature Selection

### 7.1 Feature Importance

At this stage, we intend to try to select the most effective features in training the model.

Since Random Forest is currently selected as the model, we select the important features according to this model using:

```python
model.feature_importances_
```

According to the results obtained, the importance of each feature is determined. Based on the results, V17 is the most important feature.

The feature importance values are:

```text
[0.00564158 0.00742255 0.00763433 0.01190255 0.02872176 0.00800408
 0.01718152 0.02935361 0.01325496 0.03049093 0.12849721 0.06064548
 0.12828836 0.00367727 0.14990513 0.00485518 0.06862350 0.17676839
 0.02395539 0.00622665 0.00717190 0.01699604 0.00584957 0.00445719
 0.00601504 0.00422686 0.01941772 0.01062120 0.00695445 0.00723958]
```

### 7.2 Selecting the Number of Features

Since the importance is not the only criterion and some features may have low importance but their existence is necessary for the model to perform better, at this stage, it should be done in multiple stages.

Let's try different numbers of important features on the model to find out which of the features can give the model better performance.

### 7.3 Comparison of Feature Sets

In the different tests that were conducted, almost all the results were similar, but in three cases the model performed very slightly better.

The three selected cases were:

**Top 7**

![Confusion Matrix 7](../images/FeauterSelection_Images/top7.png)

**Top 11**

![Confusion Matrix 11](../images/FeauterSelection_Images/top11.png)

**Top 20**

![Confusion Matrix 20](../images/FeauterSelection_Images/top20.png)

These results correspond to using 7, 11, and 20 features.

### 7.4 Analysis of the Results

All three have detected 70 fraudulent transactions in this test, but in the 20-feature set, the number of non-fraudulent transactions that are detected incorrectly (FP) is less than in the other two feature sets.

So, the first 20 important features are selected as the final features.



## 8. Experiment 04: Training the Model with Selected Features

### 8.1 Experiment Objective

In the previous section, we selected the first 20 important features as input features. Now, we are going to compare the cross-validation results of different models before and after feature selection.

### 8.2 Output Without Feature Selection

The cross-validation results without feature selection are as follows:


| Model | Mean Accuracy | Mean Recall | Mean Precision | Mean F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.999129 | 0.589474 | 0.869811 | 0.680979 |
| KNN Model | 0.999285 | 0.729115 | 0.848228 | 0.775246 |
| Decision Tree | 0.800666 | 0.777828 | 0.723140 | 0.642955 |
| Random Forest | 0.998957 | 0.729272 | 0.830073 | 0.737665 |

### 8.3 Output with Feature Selection

After applying feature selection and using the first 20 important features, the cross-validation results are as follows:

| Model | Mean Accuracy | Mean Recall | Mean Precision | Mean F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.999147 | 0.602105 | 0.870749 | 0.688897 |
| KNN Model | 0.999299 | 0.720761 | 0.861720 | 0.775744 |
| Decision Tree | 0.800634 | 0.752408 | 0.721476 | 0.629726 |
| Random Forest | 0.803147 | 0.743987 | 0.760287 | 0.635119 |

### 8.4 Comparison of the Results

As is clear from the results, the effect of feature selection differs between the models.

For Logistic Regression, feature selection improved all four evaluation metrics. For KNN, Accuracy, Precision, and F1 score improved slightly, while Recall decreased slightly.

For Decision Tree, Recall, Accuracy, and F1 score decreased slightly after feature selection, while Precision remained relatively similar.

For Random Forest, Recall increased from `0.729272` to `0.743987` after feature selection. This improvement in Recall means that the model was able to detect more fraudulent transactions. However, this improvement came at the cost of a decrease in Precision and F1 score.

Since the main objective of this project is fraud detection, Recall is considered the most important evaluation metric. A higher Recall means that fewer fraudulent transactions are missed by the model. Therefore, the increase in Recall for Random Forest is important for this project.

Random Forest was selected for further experiments because it was the model selected in the previous experiment based on its highest F1 score in cross-validation. In addition, after feature selection, its Recall increased from `0.729272` to `0.743987`.

Therefore, the Random Forest model with the selected 20 features was retained for the following experiments, with Recall considered the primary evaluation criterion. The decrease in Precision and F1 score will be considered in the following experiments when further improving the model.

## 9. Experiment 05: Hyperparameters

### 9.1 Experiment Objective

In this step, we want to test the models with different parameters and compare the output results.

The following parameters were tested:

* KNN → `n_neighbors = [1, 5, 20]`
* Decision Tree → `max_depth = [2, 5, 10, 7, 20, None]`
* Random Forest → `max_depth = [2, 5, 10, 7, 20, None]`
* Logistic Regression → `threshold = [0.3, 0.5, 0.6, 0.7]`
* Decision Tree → `threshold = [0.3, 0.5, 0.6, 0.7]`
* Random Forest → `threshold = [0.3, 0.5, 0.6, 0.7]`

### 9.2 KNN with Different Numbers of Neighbors

The results obtained by changing the number of neighbors are:

| Parameter Value | Precision |   Recall |       F1 |
| --------------: | --------: | -------: | -------: |
|               1 |  0.809524 | 0.715789 | 0.759777 |
|               5 |  0.956522 | 0.694737 | 0.804878 |
|              20 |  0.915493 | 0.684211 | 0.783133 |

The model performance changes with changing the number of neighbors.

When the number of neighbors is equal to 1, the model performance becomes more fluctuating. As the number of neighbors increases, the model performance becomes more stable.

Since Recall is the most important criterion in this project, `n_neighbors=1` provides the highest Recall. However, `n_neighbors=5` provides the highest F1 score and Precision.

### 9.3 Decision Tree with Different Maximum Depths

The results obtained by changing the maximum depth are:

| Parameter Value | Precision |   Recall |       F1 |
| --------------: | --------: | -------: | -------: |
|               2 |  0.835443 | 0.694737 | 0.758621 |
|               5 |  0.868421 | 0.694737 | 0.771930 |
|              10 |  0.857143 | 0.694737 | 0.767442 |
|               7 |  0.890411 | 0.684211 | 0.773810 |
|              20 |  0.800000 | 0.673684 | 0.731429 |
|            None |  0.783133 | 0.684211 | 0.730337 |

The Recall is the same for maximum depths of 2, 5, and 10. However, the highest F1 score is obtained with `max_depth=7`.

As the maximum depth increases beyond this point, the Recall decreases and the overall model performance becomes lower.

Therefore, `max_depth=7` provides the best balance between Precision, Recall, and F1 score for the Decision Tree model.

### 9.4 Random Forest with Different Maximum Depths

The results obtained by changing the maximum depth are:

| Parameter Value | Precision |   Recall |       F1 |
| --------------: | --------: | -------: | -------: |
|               2 |  0.835443 | 0.694737 | 0.758621 |
|               5 |  0.868421 | 0.694737 | 0.771930 |
|              10 |  0.857143 | 0.694737 | 0.767442 |
|               7 |  0.890411 | 0.684211 | 0.773810 |
|              20 |  0.800000 | 0.673684 | 0.731429 |
|            None |  0.783133 | 0.684211 | 0.730337 |

The results show that increasing the maximum depth does not always improve the model performance.

The highest Recall is obtained with `max_depth=2`, `5`, or `10`, while `max_depth=7` provides the highest F1 score.

Therefore, considering Recall as the main criterion, a smaller maximum depth provides a better result for this experiment.

### 9.5 Logistic Regression with Different Thresholds

The results obtained by changing the threshold are:

| Threshold | Precision |   Recall |       F1 |
| --------: | --------: | -------: | -------: |
|       0.3 |  0.835616 | 0.642105 | 0.726190 |
|       0.5 |  0.843750 | 0.568421 | 0.679245 |
|       0.6 |  0.836066 | 0.536842 | 0.653846 |
|       0.7 |  0.842105 | 0.505263 | 0.631579 |

As the threshold decreases, the Recall increases.

When the threshold changes from `0.7` to `0.3`, the Recall increases from `0.505263` to `0.642105`. This means that more fraudulent transactions are detected.

However, the Precision slightly decreases as the threshold decreases.

Since Recall is the most important criterion in this project, `threshold=0.3` provides the best result for Logistic Regression.

### 9.6 Decision Tree with Different Thresholds

The results obtained by changing the threshold are:

| Threshold | Precision |   Recall |       F1 |
| --------: | --------: | -------: | -------: |
|       0.3 |  0.868421 | 0.694737 | 0.771930 |
|       0.5 |  0.878378 | 0.684211 | 0.769231 |
|       0.6 |  0.902778 | 0.684211 | 0.778443 |
|       0.7 |  0.902778 | 0.684211 | 0.778443 |

When the threshold decreases from `0.7` to `0.3`, the Recall increases from `0.684211` to `0.694737`.

However, Precision decreases from `0.902778` to `0.868421`.

Therefore, `threshold=0.3` provides the highest Recall, while higher thresholds provide higher Precision.

Since Recall is more important in this project, `threshold=0.3` is preferred for the Decision Tree model.

### 9.7 Random Forest with Different Thresholds

The results obtained by changing the threshold are:

| Threshold | Precision |   Recall |       F1 |
| --------: | --------: | -------: | -------: |
|       0.3 |  0.878049 | 0.757895 | 0.813559 |
|       0.5 |  0.971831 | 0.726316 | 0.831325 |
|       0.6 |  0.984848 | 0.684211 | 0.807453 |
|       0.7 |  0.983871 | 0.642105 | 0.777070 |

When the threshold decreases, the Recall increases.

At `threshold=0.3`, the Recall reaches `0.757895`, which is the highest Recall among the tested thresholds. This means that more fraudulent transactions are detected.

However, Precision decreases from `0.971831` at threshold `0.5` to `0.878049` at threshold `0.3`.

The F1 score also changes with the threshold. The highest F1 score is obtained at `threshold=0.5`, with a value of `0.831325`.

## 9. Experiment 05: Hyperparameters

### 9.1 Experiment Objective

In this step, we want to test the models with different parameters and compare the output results.

The following parameters were tested:

* KNN → `n_neighbors = [1, 5, 20]`
* Decision Tree → `max_depth = [2, 5, 10, 7, 20, None]`
* Random Forest → `max_depth = [2, 5, 10, 7, 20, None]`
* Logistic Regression → `threshold = [0.3, 0.5, 0.6, 0.7]`
* Decision Tree → `threshold = [0.3, 0.5, 0.6, 0.7]`
* Random Forest → `threshold = [0.3, 0.5, 0.6, 0.7]`

### 9.2 KNN with Different Numbers of Neighbors

The results obtained by changing the number of neighbors are:

| Parameter Value | Precision |   Recall |       F1 |
| --------------: | --------: | -------: | -------: |
|               1 |  0.809524 | 0.715789 | 0.759777 |
|               5 |  0.956522 | 0.694737 | 0.804878 |
|              20 |  0.915493 | 0.684211 | 0.783133 |

The model performance changes with changing the number of neighbors.

When the number of neighbors is equal to 1, the model performance becomes more fluctuating. As the number of neighbors increases, the model performance becomes more stable.

Since Recall is the most important criterion in this project, `n_neighbors=1` provides the highest Recall. However, `n_neighbors=5` provides the highest F1 score and Precision.

### 9.3 Decision Tree with Different Maximum Depths

The results obtained by changing the maximum depth are:

| Parameter Value | Precision |   Recall |       F1 |
| --------------: | --------: | -------: | -------: |
|               2 |  0.835443 | 0.694737 | 0.758621 |
|               5 |  0.868421 | 0.694737 | 0.771930 |
|              10 |  0.857143 | 0.694737 | 0.767442 |
|               7 |  0.890411 | 0.684211 | 0.773810 |
|              20 |  0.800000 | 0.673684 | 0.731429 |
|            None |  0.783133 | 0.684211 | 0.730337 |

The Recall is the same for maximum depths of 2, 5, and 10. However, the highest F1 score is obtained with `max_depth=7`.

As the maximum depth increases beyond this point, the Recall decreases and the overall model performance becomes lower.

Therefore, `max_depth=7` provides the best balance between Precision, Recall, and F1 score for the Decision Tree model.

### 9.4 Random Forest with Different Maximum Depths

The results obtained by changing the maximum depth are:

| Parameter Value | Precision |   Recall |       F1 |
| --------------: | --------: | -------: | -------: |
|               2 |  0.835443 | 0.694737 | 0.758621 |
|               5 |  0.868421 | 0.694737 | 0.771930 |
|              10 |  0.857143 | 0.694737 | 0.767442 |
|               7 |  0.890411 | 0.684211 | 0.773810 |
|              20 |  0.800000 | 0.673684 | 0.731429 |
|            None |  0.783133 | 0.684211 | 0.730337 |

The results show that increasing the maximum depth does not always improve the model performance.

The highest Recall is obtained with `max_depth=2`, `5`, or `10`, while `max_depth=7` provides the highest F1 score.

Therefore, considering Recall as the main criterion, a smaller maximum depth provides a better result for this experiment.

### 9.5 Logistic Regression with Different Thresholds

The results obtained by changing the threshold are:

| Threshold | Precision |   Recall |       F1 |
| --------: | --------: | -------: | -------: |
|       0.3 |  0.835616 | 0.642105 | 0.726190 |
|       0.5 |  0.843750 | 0.568421 | 0.679245 |
|       0.6 |  0.836066 | 0.536842 | 0.653846 |
|       0.7 |  0.842105 | 0.505263 | 0.631579 |

As the threshold decreases, the Recall increases.

When the threshold changes from `0.7` to `0.3`, the Recall increases from `0.505263` to `0.642105`. This means that more fraudulent transactions are detected.

However, the Precision slightly decreases as the threshold decreases.

Since Recall is the most important criterion in this project, `threshold=0.3` provides the best result for Logistic Regression.

### 9.6 Decision Tree with Different Thresholds

The results obtained by changing the threshold are:

| Threshold | Precision |   Recall |       F1 |
| --------: | --------: | -------: | -------: |
|       0.3 |  0.868421 | 0.694737 | 0.771930 |
|       0.5 |  0.878378 | 0.684211 | 0.769231 |
|       0.6 |  0.902778 | 0.684211 | 0.778443 |
|       0.7 |  0.902778 | 0.684211 | 0.778443 |

When the threshold decreases from `0.7` to `0.3`, the Recall increases from `0.684211` to `0.694737`.

However, Precision decreases from `0.902778` to `0.868421`.

Therefore, `threshold=0.3` provides the highest Recall, while higher thresholds provide higher Precision.

Since Recall is more important in this project, `threshold=0.3` is preferred for the Decision Tree model.

### 9.7 Random Forest with Different Thresholds

The results obtained by changing the threshold are:

| Threshold | Precision |   Recall |       F1 |
| --------: | --------: | -------: | -------: |
|       0.3 |  0.878049 | 0.757895 | 0.813559 |
|       0.5 |  0.971831 | 0.726316 | 0.831325 |
|       0.6 |  0.984848 | 0.684211 | 0.807453 |
|       0.7 |  0.983871 | 0.642105 | 0.777070 |

When the threshold decreases, the Recall increases.

At `threshold=0.3`, the Recall reaches `0.757895`, which is the highest Recall among the tested thresholds. This means that more fraudulent transactions are detected.

However, Precision decreases from `0.971831` at threshold `0.5` to `0.878049` at threshold `0.3`.

The F1 score also changes with the threshold. The highest F1 score is obtained at `threshold=0.5`, with a value of `0.831325`.



## 10. SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) is an oversampling method used to address class imbalance by generating synthetic samples for the minority class.

Based on the results obtained from the previous experiments, Random Forest was selected for further evaluation with SMOTE. In this experiment, different values of `sampling_strategy`, maximum depth, and classification threshold were tested. The Random Forest model was evaluated with `max_depth = 5` and `7`, `max_leaf_nodes = 10`, and thresholds of `0.3`, `0.5`, and `0.6`.

Two sampling strategies were evaluated:

* `sampling_strategy = 0.06`
* `sampling_strategy = 0.1`

The main objective was to find a suitable trade-off between precision and recall while giving higher priority to recall, since correctly identifying fraudulent transactions is more important in this problem. The target was to achieve a recall of approximately 0.83 or higher while maintaining an acceptable precision.

### Threshold: 0.3, Max Depth: 5, Sampling Strategy: 0.06

| Threshold | Mean Precision | Mean Recall |  Mean F1 |
| --------- | -------------: | ----------: | -------: |
| 0.3       |       0.700471 |    0.839261 | 0.763430 |
| 0.5       |       0.762556 |    0.828690 | 0.793959 |
| 0.6       |       0.808807 |    0.811803 | 0.809721 |

### Threshold: 0.3, Max Depth: 5, Sampling Strategy: 0.1

| Threshold | Mean Precision | Mean Recall |  Mean F1 |
| --------- | -------------: | ----------: | -------: |
| 0.3       |       0.647531 |    0.841389 | 0.731568 |
| 0.5       |       0.741482 |    0.830840 | 0.783215 |
| 0.6       |       0.793746 |    0.822352 | 0.806969 |

### Max Depth: 7, Sampling Strategy: 0.06

| Threshold | Mean Precision | Mean Recall |  Mean F1 |
| --------- | -------------: | ----------: | -------: |
| 0.3       |       0.693325 |    0.839261 | 0.759100 |
| 0.5       |       0.757077 |    0.830795 | 0.791940 |
| 0.6       |       0.813267 |    0.816013 | 0.814017 |

### Max Depth: 5, Sampling Strategy: 0.1

| Threshold | Mean Precision | Mean Recall |  Mean F1 |
| --------- | -------------: | ----------: | -------: |
| 0.3       |       0.643625 |    0.841389 | 0.729055 |
| 0.5       |       0.741639 |    0.835050 | 0.785415 |
| 0.6       |       0.788814 |    0.824479 | 0.805512 |

The results show a clear trade-off between precision and recall as the classification threshold changes. At a threshold of `0.3`, the models achieve the highest recall, but this comes at the cost of a considerable decrease in precision. This means that although more fraudulent transactions are detected, the model also produces more false-positive predictions.

At a threshold of `0.6`, precision increases while recall decreases. The difference between precision and recall becomes smaller, resulting in a more balanced performance. For example, with `max_depth = 7` and `sampling_strategy = 0.06`, the model achieves a mean precision of `0.813267` and a mean recall of `0.816013`, with an F1-score of `0.814017`.

A threshold of `0.5` provides an intermediate trade-off. In several configurations, recall remains close to or above the target of `0.83 while precision is higher than at a threshold of `0.3`. Since detecting fraudulent transactions is the primary objective, maintaining a relatively high recall is more important than maximizing precision alone.

Therefore, the results indicate that a threshold of `0.5` provides a reasonable compromise between detecting fraudulent transactions and limiting false-positive predictions. However, the final model and configuration will be selected after comparing these results with the results of all previous experiments.


# 11. Final Model Selection

## 11.1 Model Selection

Based on the experimental results, Random Forest was selected as the final model. Although KNN achieved a higher F1 score in some experiments, Random Forest provided a suitable balance between Precision and Recall.

Since fraud detection is the main objective of the project, Recall was considered the primary evaluation criterion.

## 11.2 Class Weight Experiment

The effect of `class_weight='balanced'` was evaluated. Although class weighting improved Recall, it also increased False Positives and reduced Precision.

Therefore, the unweighted Random Forest model was selected.

## 11.3 Feature Selection

Random Forest feature importance was used to select the most important features. The 20-feature configuration provided a suitable balance between Recall and False Positives.

After feature selection, Recall increased from `0.729272` to `0.743987`.

Therefore, the first 20 important features were selected.

## 11.4 Hyperparameter and Sampling Experiments

Different `max_depth` values and sampling strategies were evaluated. Based on the experimental results, `max_depth=7` and `max_leaf_nodes=10` were selected.

SMOTE was also applied to address class imbalance. The selected sampling strategy was:

```python
sm = SMOTE(sampling_strategy=0.06)
```

## 11.5 Decision Threshold

Different decision thresholds were evaluated to find a suitable balance between Precision and Recall.

Although lower thresholds improved Recall, they also increased False Positives. Based on the overall experimental results, a threshold of `0.5` was selected.

## 11.6 Final Model Configuration

The final configuration was:

```python
model = RandomForestClassifier(max_depth=7, max_leaf_nodes=10)

sm = SMOTE(sampling_strategy=0.06)

threshold = 0.5
```

The final model uses:

* Random Forest
* The first 20 important features
* `max_depth=7`
* `max_leaf_nodes=10`
* SMOTE with `sampling_strategy=0.06`
* Decision threshold of `0.5`

Overall, the final configuration was selected to prioritize Recall while maintaining an acceptable level of Precision.
