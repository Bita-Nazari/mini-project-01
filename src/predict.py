import json
import pandas as pd
import joblib



model = joblib.load('./models/model.pkl')
scaler = joblib.load('./models/scaler.pkl')

def predict_data(input):
    value_df = pd.DataFrame([input])
    value_df = value_df[['V14','V10','V20','V12','V7', 'V13','V4','V15']]
    scaled_value = scaler.transform(value_df)
    treshold = 0.5
    proba = model.predict_proba(scaled_value)[: , 1]
    finalpred = (proba >= treshold).astype(int)
    str_class = ''
    if finalpred == 1:
        str_class = 'Fraud'
    else :
        str_class = "NotFraud"
    output = {'prediction' : str_class , 'class_id' :int(finalpred[0]), 'probability' :float(proba[0]) ,'threshold' : treshold ,'status' :'succes'}
    return output

if __name__=="__main__":
    json_input = json.loads(input('add your json data: '))

    output = predict_data(json_input)
    print(output)






