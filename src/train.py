from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from data_prep import get_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,f1_score,recall_score,precision_score,accuracy_score
import joblib
import pandas as pd
import numpy as np


X_train,X_test,y_train,y_test = get_data()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler,'./models/scaler.pkl')

logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled,y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)
y_pred_t_logistic = logistic_model.predict(X_train_scaled)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled,y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_t_knn = knn_model.predict(X_train_scaled)

decision_tree_model = DecisionTreeClassifier(max_depth=6)
decision_tree_model.fit(X_train_scaled,y_train)
y_pred_decision_tree= decision_tree_model.predict(X_test_scaled)
y_pred_t_decision_tree = decision_tree_model.predict(X_train_scaled)

Models = np.array((
        'Logistic Regression',
        'Knn Model',
        'Decision tree'
    ),
    dtype= str)

#region accuracy
acc_logistic_t_model = accuracy_score(y_train,y_pred_t_logistic)
acc_knn_t_model = accuracy_score(y_train,y_pred_t_knn)
acc_tree_t_model = accuracy_score(y_train,y_pred_t_decision_tree)

acc_logistic_model = accuracy_score(y_test,y_pred_logistic)
acc_knn_model = accuracy_score(y_test,y_pred_knn)
acc_tree_model = accuracy_score(y_test,y_pred_decision_tree)

accuracy_train_array = np.array((acc_logistic_t_model,acc_knn_t_model,acc_tree_t_model),dtype= float)
accuracy_test_array = np.array((acc_logistic_model,acc_knn_model,acc_tree_model),dtype=float)
acc_result_df = pd.DataFrame({
    
    'Models' :Models,
    'accurecy train' :accuracy_train_array,
    'accurecy test' : accuracy_test_array
    
})

print(acc_result_df)
#endregion
print('\n')
#region Precision
precision_logistic_t_model = precision_score(y_train,y_pred_t_logistic)
precision_knn_t_model = precision_score(y_train,y_pred_t_knn)
precision_tree_t_model = precision_score(y_train,y_pred_t_decision_tree)

precision_logistic_model = precision_score(y_test,y_pred_logistic)
precision_knn_model = precision_score(y_test,y_pred_knn)
precision_tree_model = precision_score(y_test,y_pred_decision_tree)

precision_train_array = np.array((precision_logistic_t_model ,precision_knn_t_model,precision_tree_t_model),dtype=float)
precision_test_array = np.array((precision_logistic_model ,precision_knn_model,precision_tree_model),dtype=float)

precision_result_df = pd.DataFrame({
    'Models' :Models,
    'Precision Train' : precision_train_array,
    'Precision Test' : precision_test_array
})

print(precision_result_df)

#endregion
print('\n')
#region Precision
recall_logistic_t_model = recall_score(y_train,y_pred_t_logistic)
recall_knn_t_model = recall_score(y_train,y_pred_t_knn)
recall_tree_t_model = recall_score(y_train,y_pred_t_decision_tree)

recall_logistic_model = recall_score(y_test,y_pred_logistic)
recall_knn_model = recall_score(y_test,y_pred_knn)
recall_tree_model = recall_score(y_test,y_pred_decision_tree)

recall_train_array = np.array((recall_logistic_t_model ,recall_knn_t_model,recall_tree_t_model))
recall_test_array = np.array((recall_logistic_model ,recall_knn_model,recall_tree_model))

recall_result_df = pd.DataFrame({
    'Models' : Models,
    'Recall Train' : recall_train_array,
    'Recall Test' : recall_test_array
})

print(recall_result_df)

#endregion

#region f2_score

f1_sore_logistic_t_model = f1_score(y_train,y_pred_t_logistic)
f1_sore_knn_t_model = f1_score(y_train,y_pred_t_knn)
f1_sore_tree_t_model = f1_score(y_train,y_pred_t_decision_tree)

f1_sore_logistic_model = f1_score(y_test,y_pred_logistic)
f1_sore_knn_model = f1_score(y_test,y_pred_knn)
f1_sore_tree_model = f1_score(y_test,y_pred_decision_tree)

f1_score_train_array = np.array((f1_sore_logistic_t_model ,f1_sore_knn_t_model,f1_sore_tree_t_model),dtype=float)
f1_score_test_array = np.array((f1_sore_logistic_model ,f1_sore_knn_model,f1_sore_tree_model),dtype=float)

f1_score_result_df = pd.DataFrame({
    'Models' :Models,
    'f1 Train' : f1_score_train_array,
    'f1 Test' : f1_score_test_array
})

print(f1_score_result_df)
#endregion

print('\n')
# region Confusion matrix
confution_matrix_logistic = confusion_matrix(y_train,y_pred_t_logistic)
confution_matrix_knn = confusion_matrix(y_train,y_pred_t_knn)
confution_matrix_decision_tree = confusion_matrix(y_train,y_pred_t_decision_tree)

print('Logistic : \n',confution_matrix_logistic)
print('Knn : \n',confution_matrix_knn)
print('Decision Tress : \n',confution_matrix_decision_tree)

ConfusionMatrixDisplay(confution_matrix_logistic).plot()
#plt.show()
ConfusionMatrixDisplay(confution_matrix_knn).plot()
#plt.show()
ConfusionMatrixDisplay(confution_matrix_decision_tree).plot()
#plt.show()

#endregion

