from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from data_prep import get_train_test_data,get_whole_data,get_unscaled_data
from sklearn.model_selection import cross_val_score,cross_validate,StratifiedKFold
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,f1_score,recall_score,precision_score,accuracy_score
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE 




X_train_scaled,X_test_scaled,y_train,y_test = get_train_test_data()

sm = SMOTE(sampling_strategy=0.06)

X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train)

model = RandomForestClassifier(max_depth= 7,max_leaf_nodes= 10)
model.fit(X_train_sm , y_train_sm)

joblib.dump(model,'./models/model.pkl')

y_pred_random_forest = model.predict(X_test_scaled)
