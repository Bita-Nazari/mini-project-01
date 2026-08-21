import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# region Data Preparation
dir = os.getcwd()
credit_card_df = pd.read_csv(f'{dir}/data/creditcard.csv')

print(credit_card_df.head())
credit_card_df.info()
print(credit_card_df.describe())

null_values =credit_card_df.isnull().sum().sum()
duplicate_values = credit_card_df.duplicated()
print('null values : ',null_values ,' | duplicate values: ',duplicate_values.sum())

total_count = credit_card_df['Class'].count()
not_fraud_samples = (credit_card_df['Class'] == 0).sum()
fraud_samples = (credit_card_df['Class'] == 1).sum()

not_fraud_samples_percent = (not_fraud_samples/total_count) * 100
fraud_samples_percent = (fraud_samples/total_count) * 100
print('shape : ' ,credit_card_df.shape)
print('not fraud percentage : ',not_fraud_samples_percent)
print('fraud percentage :',fraud_samples_percent)

print('not fraud Count : ',not_fraud_samples)
print('fraud Count :',fraud_samples)
print('Total Count : ',total_count)

credit_card_df['Class'].value_counts().plot(kind= 'bar')
#plt.show()

#endregion

# region Data Preprocessing
print('------------------------------------------------------------------')
print('missing values' , null_values)
print('duplicate values' , duplicate_values.sum())

credit_card_df = credit_card_df.drop_duplicates()

Q1_Amount = credit_card_df['Amount'].quantile(0.25)
Q3_Amount = credit_card_df['Amount'].quantile(0.75)

IQR_Amount = Q3_Amount - Q1_Amount
lower_Amount = Q1_Amount - 1.5*IQR_Amount
upper_Amount = Q3_Amount + 1.5*IQR_Amount

outliers =(credit_card_df['Amount'] < lower_Amount) |(credit_card_df['Amount'] > upper_Amount)
outliers_count =credit_card_df.loc[outliers, 'Amount'].count()
outliers_fraud_count = credit_card_df.loc[outliers, 'Amount'].where(credit_card_df['Class'] == 1).count()
print('outliers_count',outliers_count ,' | outliers_fraud_count : ' , outliers_fraud_count)


#Base on this information most of the amount outliers are not fraud
#but bc the total fraud count of samples in  datasets is 492,we can hypothesize amount it can be one of the importent feature for being fraud

#X = credit_card_df.drop('Class' ,axis =1)
X =credit_card_df[['V14','V10','V20','V12','V7', 'V13','V4','V15']]
y = credit_card_df['Class']

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size= 0.2,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler,'./models/scaler.pkl')

def get_train_test_data():
    return X_train_scaled,X_test_scaled,y_train,y_test

def get_unscaled_data():
    return X_train,X_test

def get_whole_data():

    return X,y




    


#endregion