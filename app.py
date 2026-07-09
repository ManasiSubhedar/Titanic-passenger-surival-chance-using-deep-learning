import streamlit as st 
import pandas as pd
from tensorflow.keras.models import load_model

import pickle

st.title("Passenger survival chance in the titanic journey")
pclass=st.slider('Enter the passenger class',1,3)
sex=st.selectbox('Enter the passenger gender', ['male','female'])
sibsp=st.slider('Enter passengers total no of  siblings and spouse',1,8)
parch=st.slider('Enter passengers total no of parents and child',0,6)
fare=st.number_input('Enter the fare of passenger')
embarked=st.selectbox("enter the passenger station from where they started their journey",['Southampton','Chebourg','Queenstown'])

data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])

model=load_model('model.h5')

with open('label_encoder.pkl','rb') as file:
    label=pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    onehot=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

data['Sex']=label.transform(data['Sex'])

embarked=onehot.transform(data[['Embarked']])



embarked = pd.DataFrame(
    embarked,
    columns=onehot.get_feature_names_out(),
    index=data.index
)
data=pd.concat([data.drop(columns='Embarked'),embarked],axis=1)

data[['Pclass','SibSp','Parch','Fare']]=scaler.transform(data[['Pclass','SibSp','Parch','Fare']])

y=model.predict(data)

y=y[0][0]

def chance(y):
    if y>0.5:
        return 'The passenger will survive the journey'
    else:
        return 'The passenger wont surive the journey'
if st.button('Predcit survial chance'):
    st.write('Probability of passenher survival chance:',y)
    st.write(data)
    st.write(data.isna().sum())
    st.write(chance(y))

print(embarked)
print(type(embarked))
print(embarked.shape)

