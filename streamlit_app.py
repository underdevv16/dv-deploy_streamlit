import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')

# st.write('Hello world!')
st.info('This app builds a machine learning model!')

with st.expander("Data:"):
  st.write('**Raw Data:**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw

with st.expander('Data Visualization:'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')


# Input Features
with st.sidebar:
  st.header('Input Features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  sex = st.selectbox('Sex', ('male', 'female'))

  # create a DataFrame for the input features:
  data = {
    'island':island,
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g,
    'sex': sex
  }
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0)


with st.expander("Input Features:"):
  st.write('*Input Penguin features:*')
  input_df
  st.write('*Combined Penguins Data:*')
  input_penguins


# Data Preparation:
# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {
  'Adelie': 0,
  'Chinstrap': 1,
  'Gentoo': 2
}

def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data Preparation:'):
  st.write('*Encoded X (Input penguin):*')
  input_row
  st.write('*Encoded y*')
  y

# Model Training and Inference:
## Train the Model
rf = RandomForestClassifier()
rf.fit(X, y)

## Apply model to make predictions
pred = rf.predict(input_row)
pred_prob = rf.predict_proba(input_row)
df_pred_prob = pd.DataFrame(pred_prob)
df_pred_prob.columns=['Adelie', 'Chinstrap', 'Gentoo']
# df_pred_prob.rename(columns={ 0: 'Adelie',
#                               1: 'Chinstrap',
#                               2: 'Gentoo'})
  
df_pred_prob
