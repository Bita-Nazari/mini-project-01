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
