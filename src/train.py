from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from data_prep import get_data
from sklearn.preprocessing import StandardScaler
import joblib


X_train,X_test,y_train,y_test = get_data()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler,'./models/scaler.pkl')

logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled,y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled,y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

decision_tree_model = DecisionTreeClassifier(max_depth=6)
decision_tree_model.fit(X_train_scaled,y_train)
y_decision_tree= decision_tree_model.predict(X_test_scaled)