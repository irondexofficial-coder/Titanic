import streamlit as st
import pickle
import pandas as pd
import numpy as np
# Import sklearn BEFORE loading the pickle file
from sklearn.ensemble import RandomForestClassifier  # or whatever model you used
from sklearn.linear_model import LogisticRegression  # add if you used this
from sklearn.tree import DecisionTreeClassifier  # add if you used this
# Add any other sklearn imports you used during training

st.title('Titanic Prediction App')
st.image('titanic.jpeg', caption='Predict Survival on the Titanic')

# Load the pretrained model
with open('titanicpickle1.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)

# Function to make predictions 
def PredictionFunction(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    try:
        prediction = model.predict([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
        return 'Survived' if prediction[0] == 1 else 'Did not Survive.'
    except Exception as e:
        return f'Something went wrong: {str(e)}. Contact IT support at +434949494.'

# Sidebar for instructions 
st.sidebar.header('How to Use!')
st.sidebar.markdown(
    """
1. Enter the Passenger Details in the Form
2. Click 'Predict' to see the survival result
3. Adjust Values to test different scenarios
    """
)
st.sidebar.info('Example: A 30 years old male, 1st class, $100 fare, traveling alone from port Southampton.')

def main():
    st.subheader('Enter Passenger Details!') 
    col1, col2 = st.columns(2)
    with col1:
        Pclass = st.selectbox('Passenger Class', options=[1, 2, 3])
        Sex = st.radio('Sex:', options=['male', 'female'])
        Age = st.slider('Age:', min_value=0, max_value=100, value=30)
    with col2:
        SibSp = st.slider('Siblings/Spouses Aboard:', min_value=0, max_value=10, value=0)
        Parch = st.slider('Parents/Children Aboard:', min_value=0, max_value=10, value=0)
        Fare = st.slider('Fare($):', min_value=0.0, max_value=500.0, step=0.01, value=50.0)
        Embarked = st.radio('Port of Embarkation:', options=['C', 'Q', 'S'])
    
    # Convert the categorical inputs to numeric values
    Sex = 1 if Sex == 'female' else 0
    Embarked = {'C': 0, 'Q': 1, 'S': 2}[Embarked]
    
    # Button for prediction
    if st.button('Predict'):
        result = PredictionFunction(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
        if 'Survived' in result:
            st.success(f'🎉 {result}')
        else:
            st.error(f'💔 {result}')

if __name__ == '__main__':
    main()
