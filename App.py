import pandas as pd
import joblib
import numpy as np
from datetime import datetime, time
import streamlit as st
from streamlit_option_menu import option_menu
#Data visualization libraries
import seaborn as sns
sns.set()
import plotly.express as px
import matplotlib.pyplot as plt

#Machine learning libraries
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

#Ignore FutureWarnings to avoid clutter in the output
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path

# Get the absolute path to the directory containing the current script
path = Path(__file__).parent.resolve()

# Add the path to sys.path if needed
import sys
sys.path.append(str(path))

# Now you can import modules from streamlit_functions_for_app
from streamlit_functions_for_app import *



#Load the trained model and scaler
model_rf=joblib.load(r"Models\model_rf.pkl")
model_svc=joblib.load(r"Models\model_svc.pkl")
model_dtc=joblib.load(r"Models\model_dtc.pkl")
model_gbc=joblib.load(r"Models\model_gbc.pkl")
model_lg=joblib.load(r"Models\model_lr.pkl")
scaler = joblib.load(r"Models\scaler.pkl")
df_orginal=pd.read_csv(r"Datas\Combined_attrition_data")
df=df_orginal.copy()

#Function to preprocess input data
def preprocess_input(data):
    #Assuming 'data' is a dictionary with input values
    df = pd.DataFrame(data, index=[0])
    label_encoder=LabelEncoder()
    #Apply the same preprocessing steps as in the training phase
    df['Over_time_Worked'] = df['Over_time_Worked'].map({'Yes': 1, 'No': 0})
    df['MaritalStatus'] = df['MaritalStatus'].map({'Single': 3, 'Married': 2, 'Divorced': 1})
    df['BusinessTravel'] = df['BusinessTravel'].map({'Non-Travel': 1, 'Travel_Rarely': 2, 'Travel_Frequently': 3})
    df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Assuming label_encoder is defined globally
    df = pd.get_dummies(df, columns=['Department', 'EducationField', 'JobRole'])
    df['Employee_Satisfaction'] = df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + df['WorkLifeBalance']
    df['Employee_rating'] = df['JobInvolvement'] + df['PerformanceRating']
    df['Employee_risk_rating'] = df[['Over_time_Worked', 'BusinessTravel', 'Employee_rating', 'MonthlyIncome',
                                     'Employee_Satisfaction', 'JobLevel', 'StockOptionLevel', 'MaritalStatus',
                                     'TrainingTimesLastYear', 'Education']].apply(lambda x:
                                                                                0 + (1 if x['MonthlyIncome'] < 30000 else 0) +
                                                                                (1 if x['BusinessTravel'] == 2 else 0) +
                                                                                (1 if x['Employee_Satisfaction'] <= 3 else 0) +
                                                                                (1 if x['MaritalStatus'] == 3 else 0) +
                                                                                (1 if x['Education'] == 1 else 0) +
                                                                                (1 if x['Employee_rating'] <= 2 else 0) +
                                                                                (1 if x['JobLevel'] == 1 else 0) +
                                                                                (1 if x['StockOptionLevel'] == 0 else 0) +
                                                                                (1 if x['TrainingTimesLastYear'] == 0 else 0) +
                                                                                (1 if x['Over_time_Worked'] == 1 else 0),
                                                                                axis=1)

    df['CombinedExperience'] = df['TotalWorkingYears'] + df['YearsAtCompany'] + df['YearsSinceLastPromotion'] + df[
        'YearsWithCurrManager']
    df.drop(columns=['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'JobInvolvement',
                     'PerformanceRating', 'TotalWorkingYears', 'YearsAtCompany', 'YearsSinceLastPromotion',
                     'YearsWithCurrManager'], inplace=True)
    col = ['EducationField_Human Resources', 'Department_Sales']

    #Check and drop each column individually
    for c in col:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    return df

#Streamlit app
def attrition_page():
    #Input features
    job_involvement = st.slider("Job Involvement", 1, 4, 2)
    performance_rating = st.slider("Performance Rating", 1, 4, 2)
    over_time_worked = st.selectbox("Over Time Worked", ["Yes", "No"])
    environment_satisfaction = st.slider("Environment Satisfaction", 1.0, 4.0, 2.0)
    job_satisfaction = st.slider("Job Satisfaction", 1.0, 4.0, 2.0)
    work_life_balance = st.slider("Work-Life Balance", 1.0, 4.0, 2.0)
    age = st.slider("Age", 18, 60, 30)
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
    distance_from_home = st.slider("Distance From Home", 1, 30, 10)
    education = st.slider("Education", 1, 5, 2)
    education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    job_level = st.slider("Job Level", 1, 5, 2)
    job_role = st.selectbox("Job Role", ["Healthcare Representative", "Manufacturing Director", "Sales Representative",
                                         "Human Resources", "Manager", "Sales Executive", "Laboratory Technician",
                                         "Research Scientist", "Research Director"])
    marital_status = st.selectbox("Marital Status", ["Married", "Divorced", "Single"])
    monthly_income = st.slider("Monthly Income", 10000, 200000, 20000)
    num_companies_worked = st.slider("Number of Companies Worked", 0, 15, 1)
    over_18 = st.selectbox("Over 18", ["Yes"])  # Assuming Over18 column is always "Yes" in the input
    percent_salary_hike = st.slider("Percent Salary Hike", 0, 50, 8)
    stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
    total_working_years = st.slider("Total Working Years", 0.0, 30.0, 1.0)
    training_times_last_year = st.slider("Training Times Last Year", 0, 10, 1)
    years_at_company = st.slider("Years at Company", 0, 20, 2)
    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
    years_with_curr_manager = st.slider("Years with Current Manager", 0, 15, 1)

    #Create a dictionary from user inputs
    input_data = {
        'JobInvolvement': job_involvement,
        'PerformanceRating': performance_rating,
        'Over_time_Worked': over_time_worked,
        'EnvironmentSatisfaction': environment_satisfaction,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'Age': age,
        'BusinessTravel': business_travel,
        'Department': department,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'EducationField': education_field,
        'Gender': gender,
        'JobLevel': job_level,
        'JobRole': job_role,
        'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income,
        'NumCompaniesWorked': num_companies_worked,
        'PercentSalaryHike': percent_salary_hike,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'YearsAtCompany': years_at_company,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
    }

    #Preprocess input data
    input_df = preprocess_input(input_data)
    model_columns=['Over_time_Worked', 'Age', 'BusinessTravel', 'DistanceFromHome',
                   'Education', 'Gender', 'JobLevel', 'MaritalStatus', 'MonthlyIncome',
                   'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel',
                   'TrainingTimesLastYear', 'Department_Human Resources',
                   'Department_Research & Development', 'EducationField_Life Sciences',
                   'EducationField_Marketing', 'EducationField_Medical',
                   'EducationField_Other', 'EducationField_Technical Degree',
                   'JobRole_Healthcare Representative', 'JobRole_Human Resources',
                   'JobRole_Laboratory Technician', 'JobRole_Manager',
                   'JobRole_Manufacturing Director', 'JobRole_Research Director',
                   'JobRole_Research Scientist', 'JobRole_Sales Executive',
                   'JobRole_Sales Representative', 'Employee_Satisfaction',
                   'Employee_rating', 'Employee_risk_rating', 'CombinedExperience']
    #Ensure feature names match those used during training
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    #Scale the input data using the saved scaler
    input_scaled = scaler.transform(input_df)
    
    prediction_rf = model_rf.predict(input_scaled)[0]
    probability_rf = model_rf.predict_proba(input_scaled)[0][1]

    prediction_svc = model_svc.predict(input_scaled)[0]
    probability_svc = model_svc.predict_proba(input_scaled)[0][1]

    prediction_dtc = model_dtc.predict(input_scaled)[0]
    probability_dtc = model_dtc.predict_proba(input_scaled)[0][1]

    prediction_gbc = model_gbc.predict(input_scaled)[0]
    probability_gbc = model_gbc.predict_proba(input_scaled)[0][1]

    prediction_lg = model_lg.predict(input_scaled)[0]
    probability_lg = model_lg.predict_proba(input_scaled)[0][1]
    
   
    prediction_rf = {0: "No Attrition (Employee will not leave)", 1: "Attrition Likely (Employee may leave)"}.get(prediction_rf, "Unknown Prediction")
    prediction_dtc = {0: "No Attrition (Employee will not leave)", 1: "Attrition Likely (Employee may leave)"}.get(prediction_dtc, "Unknown Prediction")
    prediction_gbc= {0: "No Attrition (Employee will not leave)", 1: "Attrition Likely (Employee may leave)"}.get(prediction_gbc, "Unknown Prediction")
    prediction_lg = {0: "No Attrition (Employee will not leave)", 1: "Attrition Likely (Employee may leave)"}.get(prediction_lg, "Unknown Prediction")
    prediction_svc = {0: "No Attrition (Employee will not leave)", 1: "Attrition Likely (Employee may leave)"}.get(prediction_svc, "Unknown Prediction")
    st.title(" -  -  -  -  -  -  Attrition Predictions  -  -  -  -  -  - ")
    
    col1,col2=st.columns(2)
    with col1:
        st.subheader("Random Forest Model")
        st.write(f"Prediction: :blue[**{prediction_rf}**]")
        st.write(f"Probability of Attrition: :blue[**{probability_rf:.2%}**]")

        st.subheader("Decision Tree Classifier")
        st.write(f"Prediction: :orange[**{prediction_dtc}**]")
        st.write(f"Probability of Attrition: :orange[**{probability_dtc:.2%}**]")

        st.subheader("Gradient Boosting Classifier")
        st.write(f"Prediction: :blue[**{prediction_gbc}**]")
        st.write(f"Probability of Attrition: :blue[**{probability_gbc:.2%}**]")
    with col2:
        st.subheader("Logistic Regression")
        st.write(f"Prediction: :orange[**{prediction_lg}**]")
        st.write(f"Probability of Attrition: :orange[**{probability_lg:.2%}**]")

        st.subheader("Support Vector Classifier")
        st.write(f"Prediction: :blue[**{prediction_svc}**]")
        st.write(f"Probability of Attrition: :blue[**{probability_svc:.2%}**]")

def about_page():
    st.title(":blue[**Attrition Alchemy: Data-Driven Insights and Predictive Strategies for Employee Retention**] ")
    col1, col2 = st.columns([4,2], gap='medium')
    with col1:
        st.subheader(":violet[Problem Statement:]")
        st.write("""
                      A large company named XYZ, employs,at any given point of time, around 4000 employees.
                      However, every year, around 15% of its employees leave the company and need to be replaced with the talent pool available in the job market.
                      The management believes that this level of attrition (employees leaving, either on their own or because they got fired) is bad for the company,
                      because of the following reasons
                      - The former employees’ projects get delayed, which makes it difficult to meet timelines, resulting in a reputation loss among consumers and partners.
                      - A sizeable department has to be maintained, for the purposes of recruiting new talent.
                      - More often than not, the new employees have to be trained for the job andor given time to acclimatise themselves to the company.
                      - Hence, the management has contracted an HR analytics firm to understand what factors they should focus on, in order to curb attrition. 
                      - In other words,they want to know what changes they should make to their workplace,in order to get most of their employees to stay.
                      Also, they want to know which of these variables is most important and needs to be addressed right away.""")
        st.subheader(":violet[Objective:]")
        st.write("""
                 - The main objective of this Employee Attrition Prediction App is to help organizations proactively identify employees 
                 who are at risk of leaving the company.
                 - By predicting attrition, businesses can take preventive measures to retain valuable talent."""
        )

        st.subheader(":violet[Approaches:]")
        st.write("""
                 - The app utilizes machine learning models, including Random Forest, Decision Tree,Gradient Boosting, 
                 Logistic Regression, and Support Vector Classifier, to make predictions.
                 - These models are trained on historical employee data, considering various factors such as age,salary,
                  job role, and work-related factors."""
        )

        st.subheader(":violet[Tools and Technologies Used:]")
        st.write("""
                 The following tools and technologies were used in the development of this app:
                 - Streamlit: A Python library for creating web apps with minimal effort.
                 - Scikit-learn: A machine learning library for building and training predictive models.
                 - Joblib: A library for saving and loading machine learning models efficiently.
                 - Pandas: A data manipulation library for handling structured data.
                 - Matplotlib, Seaborn, Plotly: Data visualization libraries for creating informative charts.
                 - Imbalanced-learn: A library for handling imbalanced datasets, used for oversampling with SMOTE."""
        )

        st.subheader(":violet[Business Domain]")
        st.write("""
                 - The app is designed for businesses operating in various domains that want to address employee attrition challenges.
                 - It can be particularly beneficial for human resources,management, and leadership teams to 
                 make data-driven decisions in talent retention."""
        )

    
    with col2:
        col2.markdown("#   ")
        col2.markdown("## <style>h2 {font-size: 18px;}</style> :orange[Goal of the Project] : ", unsafe_allow_html=True)
        col2.markdown("""
                      ## <style>h2 {font-size: 16px;}</style> 
                      - Build a Data model.
                      - Build an Executive Dashboard to present insights on the Attrition.
                      - Build a prediction model (Min 3) and compare all the three models.
                      """, unsafe_allow_html=True)
        col2.markdown("#   ")
        col2.markdown("#   ")
        col2.markdown("#   ")
        col2.markdown("""
                      ## <span style="font-size: 18px;">:orange[Created By]:</span><br>
                      Raghavendran S,<br>
                      Data Scientist Aspirant,<br>
                      email-id: [raghavendranhp@gmail.com](mailto:raghavendranhp@gmail.com)<br>
                      [LinkedIn-Profile](https://www.linkedin.com/in/raghavendransundararajan/),<br>
                      [GitHub-Link](https://github.com/raghavendranhp)
                        """, unsafe_allow_html=True)
       
def insight_page():
    question_dictionary={
'Attrition Count Analysis':1,
'Distribution of Age':2,
'Attrition Based on Job Level':3,
'Attrition Count by Gender and Marital Status':4,
'Attrition Count by Employee Stasification':5,
'Attrition Count based on Overtime Work of Employee':6,
'Distribution of Monthly Income':7,
'Attrition Based on Job role':8,
'Attrition Based on Business Travel':9,
'Count of People Spending Years with a Manager in an Organization':10,
'Recommendations':11}
    #Grouping the DataFrame 'df' by 'Attrition', 'Gender', and 'MaritalStatus' and calculating the count of each group
    gender_atr_df = df.groupby(['Attrition', 'Gender', 'MaritalStatus']).size().reset_index(name='Count')
    #Selecting rows where 'Attrition' is 'Yes' to create a DataFrame for employees with attrition
    gen_attrition_yes_df = gender_atr_df[gender_atr_df['Attrition'] == 'Yes']
    #Calculate the mean MonthlyIncome for each JobRole
    income = df.groupby(by='JobRole').MonthlyIncome.mean()
    income_df_jobrole = pd.DataFrame(income)
    income_df_jobrole = income_df_jobrole.sort_values(by='MonthlyIncome')
    #Get JobRoles with Attrition 'Yes' and their counts
    jobrole_attrition = df[df['Attrition'] == 'Yes']['JobRole']
    jobrole_atr_value_counts = jobrole_attrition.value_counts()
    jobrole_atr_df = pd.DataFrame(jobrole_atr_value_counts)

    #Create a copy of the DataFrame
    df1 = df.copy()
    #Map education levels to descriptive labels
    df1['Education'] = df1['Education'].map({5: "Doctrate", 4: "Master's", 3: "Bachelor's", 2: "College", 1: "Below_college"})
    #Calculate the mean MonthlyIncome for each Education level
    edu_sal = df1.groupby('Education').MonthlyIncome.mean().round()
    edu_sal_df = pd.DataFrame(edu_sal)
    edu_sal_df = edu_sal_df.sort_values('MonthlyIncome', ascending=False)
    
    questions=list(question_dictionary.keys())
    question_option=st.selectbox(
    'Select the Feature Analysis',
    (questions))
    required_function=question_dictionary[question_option]
    if required_function==1:
        display_attrition_count(df)
    elif required_function==2:
        display_age_distribution(df)
    elif required_function==3:
        display_job_level(df)
    elif required_function==4:
        display_gender_maritalstatus_attrition(gen_attrition_yes_df)
    elif required_function==5:
        display_employee_satsification(df)
    elif required_function==6:
        display_overtime_worked(df)
    elif required_function==7:
        display_attrition_monthly_income(df)
    elif required_function==8:
        display_jobrole_attrition(jobrole_atr_df)
    elif required_function==9:
        display_business_travel(df)
    elif required_function==10:
        display_years_with_currmanager(df)
    elif required_function==11:
        display_recommendations()


def main():
    st.set_page_config(page_title="Attrition Alchemy | By Raghavendran",layout="wide",page_icon="🧊")
    selected = option_menu('',["About","App","Explore"],
                           icons=["house","graph-up-arrow","bar-chart-line"],
                           menu_icon="menu-button-wide",
                           default_index=0,orientation="horizontal",
                           styles={"nav-link": {"font-size": "18px", "text-align": "left", "margin": "-2px", "--hover-color": "#FF5A5F"},
                                   "nav-link-selected": {"background-color": "#6495ED"}}
                           )
    #Aboutpage
    if selected == "About":
        about_page()

    #App Page
    elif selected == "App":
        st.header(":violet[**Employee Attrition Prediction App**] -created by Raghav")
        attrition_page()
    
    #Insights Page
    elif selected == "Explore":
        st.header(":violet[**Employee Attrition -Insights**]")
        insight_page()

#Run the Streamlit app
if __name__ == "__main__":
    main()
