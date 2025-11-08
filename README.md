# Credit-Card-Fraud-Detection-System
From Raw Data -> Business Insights -> Automation

## Table of Contents

* [Project Overview](#project-overview)
* [Project Objectives](#project-objectives)
* [Tools & Technologies Used](#tools--technology-used)
* [Project Workflow](#project-workflow)
* [Excel Dashboard](#excel-dashboard)
* [Key Insights & Results](#key-insights--results)
* [Dataset Source](#dataset-source)

### Project Overview
This project was developed as part of my Industrail Training (IT) in Data Analytics and Machine Learning.
The goal was to build an intelligent system that detects fraudulent credit card transactions based on transaction data.

The project combines Data Analytics, Machine Learning, and Model Deployment techniques to create a
full end-to-end solution - from data preprocessing to model visualization and web deployment.

### Project Objectives
* To analyze credit card transaction data and uncover patterns between fraudulent and legitimate transactions.
* To build a predictive model that accurately classifies transactions as fraudulent or legitimate.
* To deploy the trained model using Streamlit for real-time predictions.

### Tools & Technologies Used
| Category | Tools |
|----------|-------|
| Languages | Python |
| Libraries | Pandas, Numpy, Scikit-learn, Matplotlib, Seaborn, Joblib |
| Deployment | Streamlit |
| Data Cleaning | Excel, Power Query|
| IDE | Spyder, VS Code |

### Project Workflow
The project was divided into four main phases:
#### 1. Data Collection & Preprocessing:
* Loaded the dataset using Excel and Python (Pandas, Numpy).
* Cleaned the data using power query and loaded it back to excel.
* Handled missing & duplicate values, converted datatype and encoded categorical features.
* Scaled numerical features.
```python
# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Loading the Dataset
dataset = pd.read_excel('credit card fraud detection dataset.xlsx')
# Checking data Info
print(dataset.info())
print(dataset.describe())
# Data Preprocessing
# Handling Missing and Duplicate Values
dataset = dataset.dropna()
dataset = dataset.drop_duplicates()
# Datatype Conversion
dataset['trans_time'] = pd.to_timedelta(dataset['trans_time'])
dataset['trans_time_hrs'] = pd.to_numeric(dataset['trans_time_hrs'])
# Encoding Categorical Features
dataset = pd.get_dummies(dataset, columns=['category'], drop_first=True, dtype=int)
print(dataset.info())
# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataset['amt'] = scaler.fit_transform(dataset[['amt']])
dataset['trans_time'] = scaler.fit_transform(dataset[['trans_time']])
dataset['trans_time_hrs'] = scaler.fit_transform(dataset[['trans_time_hrs']])
```

<img width="960" height="504" alt="fraud detection preprocessed data_excel" src="https://github.com/user-attachments/assets/2ee96ed9-d070-4108-8bbd-7fa92a5c2401" />

#### 2. Exploratory Data Analysis:
* Analyzed transaction trends, correlations, and class distributions via excel (using pivot tables) and python.
* Created visualizations using Matplotlib and seaborn to understand fraud occurrence patterns.
* Identified correlations with target variable.
```python
# Visualize correrlation between features
plt.figure(figsize=(10,8))
sns.heatmap(dataset.corr(), annot=None, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Feature Selection
# Identifyng correlation with target variable
corr_matrix = dataset.corr()
important_features = corr_matrix['is_fraud'].sort_values(ascending=False)
print(important_features)
```
<img width="960" height="505" alt="fraud detection descriptive analytics" src="https://github.com/user-attachments/assets/e415384d-bbe2-4e50-af6c-eea24c918ee7" />
<img width="712" height="637" alt="Feature Correlation Heatmap" src="https://github.com/user-attachments/assets/0f222574-828e-4d24-99f0-9abbe5820924" />

#### 3. Model Development & Evaluation:
* Splitted into train and test sets and handled data imbalance.
* Trained a Logistic Regression Model and handled over-fitting issue.
* Evaluated model using accuracy, precision, recall, and F1-score.
* Saved trained model using joblib for deployment.
```python
# Model Selection and Training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Splitting the Dataset
x = dataset.drop('is_fraud', axis=1)
y = dataset['is_fraud']
# Splitting into Train and Test Set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# Handling Data Imbalance
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_train_balanced).value_counts())
# Training the Model
model = LogisticRegression(penalty='l1', C=10, random_state=0, solver='liblinear')
model.fit(x_train_balanced, y_train_balanced)

# Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Predicting the Test Set Results
y_pred = model.predict(x_test)
# Evaluate Performance
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
accuracy_score(y_test, y_pred)

# Saved Trained Model
import joblib
joblib.dump(model, 'credit_card_fraud_detection_model.pkl')
```
<img width="379" height="282" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/c103760c-ddae-4c67-bd22-34f89211cf18" />

#### 4. Model Deployment:
* Deployed the model using Streamlit to allow users to interact with the model through a simple web interface.
* Users can input transaction details and get instant predictions.
```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load('credit card fraud detection model.pkl')
scaler = joblib.load('scaler.pkl')

# App Title
st.title("Credit Card Fraud Detection System")
st.write("This web app predicts whether a transaction is **Fraudulent** or **Legitimate** using a trained Machine Learning Model.")

# Create user inputs
st.header("Enter Transaction Details")

# Input fields
credit_card_number = st.text_input("Credit Card Number", max_chars=16, 
                                   placeholder="1234 5678 9012 3456")
if credit_card_number:
    cc_clean = credit_card_number.replace(" ", "")
    
    if not cc_clean.isdigit():
        st.error("Please enter only digits")
    else:
        st.success("Card Format looks good")
        cc_numeric = int(cc_clean)
amount = st.number_input("Transaction Amountd($)", min_value=0.0, value=0.00)
time = st.number_input("Transaction Time (hours)", min_value=0, max_value=23)
is_night_transaction = st.number_input("Transaction by Night", min_value=0, max_value=1)

# Create Dataframe with same structure as training data
sample_data = pd.DataFrame([[credit_card_number, amount, time, is_night_transaction]], 
                           columns=['cc_num', 'amt', 'trans_time_hrs', 'trans_time_is_night'])

# Scale input data
scaled_data = scaler.transform(sample_data)

# Predict
if st.button("Predict Transaction"):
    prediction = model.predict(scaled_data)
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Legitimate Transaction.")
        
st.markdown("---")
st.caption("Developed by Okafor Gift Chukwudi - IT Industrial Training Project on Data Analytics & Machine Learning.")
```
<img width="747" height="767" alt="screenshot-1762633770035" src="https://github.com/user-attachments/assets/5ecfb998-bf1f-4472-90a0-cb7dea19f850" />

### Excel Dashboard
The Excel Dashboard includes the following visuals:
* 🆚 Fraud vs Non-Fraud transactions.
* 💰Transaction amount by class.
* 🗺️Geolocation Mismatch.
* ⏲️Fraud occurence over time.
* 💰⏲️Transaction amount vs time.
* 🌃Night/Weekend transaction by class.

<img width="960" height="504" alt="credit card fraud insight dashboard" src="https://github.com/user-attachments/assets/46981d55-e86b-4c98-8f19-683bf23511e8" />

### Key Insights & Results
* The dataset was highly imbalanced, with fraudulent transactions representing 1.05% of total records.
* Legitimate transactions recorded high maximium transactions compared to the fraudulent transactions.
* More transactions were made relatively far from the merchant's location.
> 42994 transactions were made relatively far from the merchant's location,
> and 432 of those transactions were fraudulent.
* More fraud occurrences were recorded between 10PM and 11PM.
* Peak rise of transaction amount over time totaling up to $275,533.99 as at 11PM.
* High amount of transactions were recorded at night and during the weekend for legitimate cases compared to fraudulent cases.
* The trained model achieved 94.0% accuracy, effectively distinguishing fraudulent transactions.
* The final dashboard and web app made it easy to visualize insights and model predictions.

### Dataset Source

[Download Here](https://www.kaggle.com/datasets/orogunadebola/credit-card-transaction-dataset-fraud-detection)
