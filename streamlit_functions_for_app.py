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

def display_attrition_count(df):
    #Set a custom color palette
    colors = ["#4CAF50", "#FFC107"]  # Green for 'No', Amber for 'Yes
    #Set the figure size
    plt.figure(figsize=(5, 3))
    #Use a horizontal countplot with custom colors
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Attrition', orient='h', palette=colors, hue='Attrition', ax=ax)
    sns.despine(left=True)
    #Set labels and font sizes
    ax.set_ylabel('Count', fontsize=10)
    ax.set_xlabel('Attrition', fontsize=10, rotation=0, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    #Display the plot
    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate('{:}'.format(p.get_height()), (x.mean(), y - 150), ha='center', va='bottom', fontsize=10, color='black')
    #Show the plot in Streamlit
    st.pyplot(fig)
    #Display insights in a box
    st.info("## Insights:")
    st.info("- The majority of employees in the dataset did not experience attrition ('No').")
    st.info("- The count of employees with attrition ('Yes') is relatively lower.")
    st.info("- Further exploration into the factors influencing attrition may be necessary.")

def display_age_distribution(df):
    #Set the figure size
    fig, ax = plt.subplots(figsize=(6, 4))
    #Plot the distribution of Age where attrition is false
    sns.histplot(df[df['Attrition'] == 'No']['Age'], label='Non Attrition', kde=False, bins=10)
    #Plot the distribution of Age where attrition is true
    sns.histplot(df[df['Attrition'] == 'Yes']['Age'], label='Attrition', kde=False, bins=10)
    #Remove the spines on the left
    sns.despine(left=True)
    #Remove the vertical gridlines
    plt.grid(axis='x')
    #Set labels and font sizes
    plt.xlabel('Age', fontsize=10)
    plt.ylabel('Density', fontsize=10, rotation=0, labelpad=30)
    plt.title('Distribution of Age', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    #Adjust the legend size and position
    plt.legend(fontsize='small', bbox_to_anchor=(0.03, 0.95), loc=2, borderaxespad=0., frameon=1)
    #Show the plot in Streamlit
    st.pyplot(fig)
    #Display insights below the figure
    st.info("## Insights:")
    st.info("- The distribution of age for employees without attrition is concentrated in the range of 40 to 60.")
    st.info("- Employees with attrition show a broader age distribution, with a peak in the 25 to 35 range.")
    st.info("- Further analysis may be needed to understand the age-related factors contributing to attrition.")


def display_job_level(df):
    #Set the figure size
    fig, ax = plt.subplots(figsize=(6, 4))
    #Use Seaborn's countplot to show the count of each 'JobLevel' category, differentiated by 'Attrition'
    sns.countplot(x=df['JobLevel'], data=df, hue="Attrition", ax=ax)
    #Add labels to the bars in the countplot
    for container in ax.containers:
        ax.bar_label(container)
    #Set plot title and axis labels
    plt.title('JobLevel', backgroundcolor='black', color='white', fontsize=10)
    plt.xlabel('JobLevel', fontsize=10)
    #Display grid lines
    plt.grid()
    #Show the plot in Streamlit
    st.pyplot(fig)
    #Display insights below the figure
    st.info("## Insights:")
    st.info("- Job Level 1 and 2 have a higher count of employees with attrition ('Yes').")
    st.info("- Job Level 3,4,5 shows a relatively higher count of employees without attrition ('No').")
    st.info("- Further analysis may be required to understand the factors contributing to attrition at different job levels.")

def display_gender_maritalstatus_attrition(gen_attrition_yes_df):
    #Set the figure size
    fig, ax = plt.subplots(figsize=(6, 4))
    #Use Seaborn's barplot to create a grouped bar plot
    sns.barplot(x='Gender', y='Count', hue='MaritalStatus', data=gen_attrition_yes_df, ax=ax)
    #Add labels and title
    plt.xlabel('Gender and Marital Status') 
    plt.ylabel('Count')
    plt.title('Attrition Count by Gender and Marital Status')
    #Show the plot in Streamlit
    st.pyplot(fig)
    #Display insights below the figure
    st.info("## Insights:")
    st.info("- The grouped bar plot illustrates the count of attrition for different gender and marital status combinations.")
    st.info("- For example, among males, singles individuals have a higher attrition count compared to married.")
    
def display_employee_satsification(df):
    #Filtering DataFrame to include only rows where 'Attrition' is 'Yes'
    x_yes = df[df['Attrition'] == 'Yes']
    #Calculating 'Employee_Satisfaction' by summing 'JobSatisfaction', 'EnvironmentSatisfaction', and 'WorkLifeBalance'
    x_yes['Employee_Satisfaction'] = x_yes['JobSatisfaction'] + x_yes['EnvironmentSatisfaction'] + x_yes['WorkLifeBalance']
    #Pie chart for Job Satisfaction distribution of employees with Attrition (Yes)
    y1 = x_yes['JobSatisfaction'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(y1, labels=['4', '3', '1', '2'], autopct='%1.1f%%')
    ax1.legend(loc='lower left')
    ax1.set_title('Job Satisfaction of Employees - Attrition (Yes)')
    #Pie chart for Environment Satisfaction distribution of employees with Attrition (Yes)
    y2 = x_yes['EnvironmentSatisfaction'].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(y2, labels=['4', '3', '2', '1'], autopct='%1.1f%%')
    ax2.legend(loc='lower left')
    ax2.set_title('Environment Satisfaction of Employees - Attrition (Yes)')
    #Pie chart for WorkLifeBalance distribution of employees with Attrition (Yes)
    y3 = x_yes['WorkLifeBalance'].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.pie(y3, labels=['4', '3', '2', '1'], autopct='%1.1f%%')
    ax3.legend(loc='lower left')
    ax3.set_title('WorkLifeBalance of Employees - Attrition (Yes)')
    #Creating a new DataFrame 'y4' and calculating 'Employee_Satisfaction' for all employees
    y4 = df.copy()
    y4['Employee_Satisfaction'] = y4['JobSatisfaction'] + y4['EnvironmentSatisfaction'] + y4['WorkLifeBalance']
    #Countplot for Employee Satisfaction distribution for all employees
    fig4, ax4 = plt.subplots()
    sns.countplot(x=y4['Employee_Satisfaction'], data=y4, hue="Attrition", ax=ax4)
    for container in ax4.containers:
        ax4.bar_label(container)
    ax4.set_title('Employee Satisfaction')
    ax4.set_xlabel('Employee Satisfaction')
    ax4.grid()
    #Display plots in Streamlit
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)
    st.pyplot(fig4)
    #Display insights below the plots
    st.info("## Insights:")
    st.info("- The pie charts show the distribution of Job Satisfaction, Environment Satisfaction, and WorkLifeBalance for employees with Attrition (Yes).")
    st.info("- The countplot illustrates the Employee Satisfaction distribution for all employees, differentiated by Attrition status.")

def display_jobrole_attrition(jobrole_atr_df):
    fig, ax = plt.subplots(figsize=(8, 3))
    #Use Seaborn's barplot to create a bar plot of attrition count per job role
    sns.barplot(y=jobrole_atr_df.index, x='count', data=jobrole_atr_df,
                color="lightcoral", ci=None, ax=ax)
    #Set plot title and labels
    ax.set_title('Attrition Count per Job Role')
    ax.set_xlabel('Attrition Count')
    ax.set_ylabel('Job Role')
    plt.xticks(rotation=0)
    plt.yticks()
    #Show the plot in Streamlit
    st.pyplot(fig)
    #Display insights below the plot
    st.info("## Insights:")
    st.info("- The bar plot illustrates the count of attrition for each job role.")
    st.info("- Sales Executives and Research Scientists have higher attrition counts compared to Managers.")

def display_overtime_worked(df):
    #Set the figure size
    fig, ax = plt.subplots(figsize=(5, 3))
    #Use Seaborn's countplot to create a bar plot of employees by 'Over_time_Worked' and 'Attrition' status
    sns.countplot(x='Over_time_Worked', hue='Attrition', data=df, ax=ax)
    #Add count annotations on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
    #Set labels and title
    ax.set_xlabel('Attrition', fontsize=10)
    ax.set_ylabel('Count', fontsize=10, rotation=0, labelpad=30)
    ax.set_title('Count of Employees by Over_time_Worked and Attrition Status', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Attrition', bbox_to_anchor=(1, 1), loc='upper left')
    #Show the plot in Streamlit
    st.pyplot(fig)
    #Display insights below the plot
    st.info("## Insights:")
    st.info("- The countplot shows the distribution of employees based on 'Over_time_Worked' and 'Attrition' status.")
    st.info("- Employees who worked overtime ('Yes') seem to have a higher attrition count.")
    
def display_attrition_monthly_income(df):
    #Set the figure size
    fig, ax = plt.subplots(figsize=(8, 3))
    #Create a histogram using histplot for Non-Attrition
    sns.histplot(df[df['Attrition'] == 'No']['MonthlyIncome'], label='Non Attrition', kde=False, bins=10, color='skyblue', ax=ax)
    #Create a histogram using histplot for Attrition
    sns.histplot(df[df['Attrition'] == 'Yes']['MonthlyIncome'], label='Attrition', kde=False, bins=10, color='salmon', ax=ax)
    #Remove the spines on the left
    sns.despine(left=True)
    #Remove the vertical gridlines
    plt.grid(axis='x')
    #Set labels and font sizes
    plt.xlabel('Monthly Income', fontsize=18)
    plt.ylabel('Density', fontsize=10, rotation=0, labelpad=30)
    plt.title('Distribution of Monthly Income', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    #Adjust the legend size and position
    plt.legend(fontsize='x-large', bbox_to_anchor=(0.4, 0.94), loc=2, borderaxespad=0., frameon=0)
    #Display the average monthly income information
    plt.axvline(df['MonthlyIncome'].mean(), color='green', linestyle='dashed', linewidth=2, label='Overall Mean')
    plt.axvline(df[df['Gender'] == 'Male']['MonthlyIncome'].mean(), color='blue', linestyle='dashed', linewidth=2, label='Male Mean')
    plt.axvline(df[df['Gender'] == 'Female']['MonthlyIncome'].mean(), color='purple', linestyle='dashed', linewidth=2, label='Female Mean')
    #Show the plot in Streamlit
    st.pyplot(fig)
    #Display insights below the plot
    st.info("## Insights:")
    st.info("- The histogram shows the distribution of Monthly Income for employees with and without attrition.")
    st.info("""- The dashed lines represent the Overall Average Monthly Income: 65029.0,
            Average Monthly Income for Males: 65319.0,
            Average Monthly Income for Females: 64595.0.""")
    st.info("- The attrition is high for lower monthly income.")
    
def display_business_travel(df):
    #Set the figure size
    fig, ax = plt.subplots(figsize=(5, 3))
    #Create a count plot using Seaborn
    sns.countplot(x='BusinessTravel', hue='Attrition', data=df, palette=["#7FFF00", "#458B00"], ax=ax)
    #Set x-axis label, y-axis label, and plot title
    ax.set_xlabel('Business Travel', fontsize=10)
    ax.set_ylabel('Count', fontsize=10, rotation=0, labelpad=30)
    ax.set_title('Count of Employees by Business Travel and Attrition Status', fontsize=10)
    #Show the plot in Streamlit
    st.pyplot(fig)
    #Display insights below the plot
    st.info("## Insights:")
    st.info("- The count plot illustrates the distribution of employees based on 'Business Travel' and 'Attrition' status.")
    st.info("- Employees who travel rarely seem to have a higher count of non-attrition.")

def display_years_with_currmanager(df):
    #Group and count by 'YearsWithCurrManager' and 'Attrition'
    man_att = df.groupby(['YearsWithCurrManager', 'Attrition']).apply(lambda x: x['MonthlyIncome'].count()).reset_index(name='Counts')
    #Create a line plot using Plotly Express
    fig = px.line(man_att, x='YearsWithCurrManager', y='Counts', color='Attrition',
                title='Count of People Spending Years with a Manager in an Organization')
    #Show the plot in Streamlit
    st.plotly_chart(fig)
    #Display insights below the plot
    st.info("## Insights:")
    st.info("- The line plot visualizes the count of people spending years with a manager, differentiated by attrition status.")
    st.info("- The plot may indicate trends in attrition based on the number of years spent with the current manager.")
    
def display_recommendations():
    #Display recommendations using HTML/CSS
    st.markdown("""
    <h3 align="lefAt"><font color=brown> Recommendations:</font></h3>

- The below recommendations is based on the key findings related to reducing attrition rate.<br>


1. Age:<br>
    - Implement strategies to address the specific needs and career aspirations of employees across different age groups.
    - This can include offering targeted development opportunities, mentorship programs, and flexible work arrangements to support work-life balance.<br>

    
2. Compensation:<br>
    - Regularly review and benchmark compensation packages to ensure they are competitive in the market.<br>
    - Consider incorporating performance-based incentives and rewards to motivate employees and recognize their contributions.<br>

    
3. Job experience:<br>
    - Provide opportunities for career advancement, skill development, and cross-functional training.<br>
    - Establish clear career paths and provide regular feedback and performance evaluations to support employee growth and engagement.<br>

    
4. Specific job-related variables:
    - Tailor retention strategies based on different job roles and responsibilities.<br>
    - This can include improving job satisfaction, providing challenging assignments, and fostering a positive work environment.

    
5. Job-related aspects:<br>
    - Enhance employee engagement and job satisfaction by offering a supportive work environment.<br>
    - Provide opportunities for professional development, promote a culture of continuous learning, and ensure fair and transparent processes for promotions and career growth.<br>

    
6. Work-related factors:<br>
    - Focus on improving factors such as environment satisfaction, job involvement, job satisfaction, work-life balance, and managing overtime demands.<br>
    - Conduct regular employee surveys to understand their concerns and feedback, and take proactive measures to address any identified areas of improvement.<br>

    
7. Overall:<br>
    - Foster a positive organizational culture that values employee well-being, work-life balance, and growth opportunities.<br> 
    - Encourage open communication, provide avenues for feedback and suggestions, and regularly evaluate and refine retention strategies based on employee feedback and changing needs.<br></div>
""",unsafe_allow_html=True)
