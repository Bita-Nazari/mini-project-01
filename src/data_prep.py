import pandas as pd
import matplotlib.pyplot as plt
import os

dir = os.getcwd()
credit_card_df = pd.read_csv(f'{dir}/data/creditcard.csv')

print(credit_card_df.head())
credit_card_df.info()
print(credit_card_df.describe())

null_values =credit_card_df.isnull().sum().sum()
duplicate_values = credit_card_df.duplicated().sum()
print('null values : ',null_values ,' | duplicate values: ',duplicate_values)

total_count = credit_card_df['Class'].count()
not_fraud_samples = (credit_card_df['Class'] == 0).sum()
fraud_samples = (credit_card_df['Class'] == 1).sum()

not_fraud_samples_percent = (not_fraud_samples/total_count) * 100
fraud_samples_percent = (fraud_samples/total_count) * 100
print('shape : ' ,credit_card_df.shape)
print('not fraud percentage : g',not_fraud_samples_percent)
print('fraud percentage :',fraud_samples_percent)

credit_card_df['Class'].value_counts().plot(kind= 'bar')
plt.show()