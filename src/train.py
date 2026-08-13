from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from data_prep import get_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score
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
    
    'Models' :[
        'Logistic Regression',
        'Knn Model',
        'Decision tree'
    ],
    'accurecy train' :accuracy_train_array,
    'accurecy test' : accuracy_test_array
    
})

print(acc_result_df)

#endregion