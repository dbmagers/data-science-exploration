#!
"""
Created on Fri Feb 21 11:53:01 2025

@author: Brandon Magers
Analysis of HR dataset to predict attrition and monthly income
"""

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from dmba import classificationSummary, regressionSummary, adjusted_r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function for preprocessing data into category data types
def get_categories(df):
    # Set columns of category type of string vars with dummies
    df = pd.get_dummies(df, columns=['BusinessTravel', 'Department', 'Gender', 'EducationField',
                                                         'JobRole', 'MaritalStatus', 'OverTime'], drop_first=True)
    # Set columns to category type (ordered)
    scale4_dtype = CategoricalDtype(categories=[1, 2, 3, 4], ordered=True) # 4 scale category
    list_temp = ['EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction',
                 'RelationshipSatisfaction', 'WorkLifeBalance']
    df[list_temp] = df[list_temp].astype(scale4_dtype)
    df['Education'] = pd.Categorical(df['Education'], categories=[1, 2, 3, 4, 5], ordered=True)
    df['JobLevel'] = pd.Categorical(df['JobLevel'], categories=[1, 2, 3, 4, 5], ordered=True)
    df['StockOptionLevel'] = pd.Categorical(df['StockOptionLevel'], categories=[0, 1, 2, 3], ordered=True)
    return df

# Fuction to run logistic regression on attrition outcome
def run_attrition(df_attrition):
    # print heading
    print('*'*60)
    print('*'+' '*58+'*')
    print('*'+' '*14+'Attrition - Logistic Regression'+' '*13+'*')
    print('*'+' '*58+'*')  
    print('*'*60)
        # Attrition prediction data preprocessing
    df_attrition['MonthlyIncome'] = df_attrition['MonthlyIncome'] / 1000 
    df_attrition = get_categories(df_attrition)
    df_attrition.drop(columns=['JobLevel'], inplace=True) # high correlation with monthly income
    # Print new info and corr heatmap after preprocessing
    print(df_attrition.info())
    plt.figure(figsize=(5, 4))
    sns.heatmap(df_attrition.corr(), cmap='coolwarm')
    plt.show()
    
    ## Predict attrition with logistic
    x = df_attrition.drop(columns=['Attrition', 'Gender_Male', 'NumCompaniesWorked',
                                   'RelationshipSatisfaction', 'YearsSinceLastPromotion',
                                   'Department_Research & Development', 'Department_Sales',
                                   'Education', 'PercentSalaryHike']) # found to have little impact on model accuracy
    y = df_attrition['Attrition']
    # Split data into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    
    # Build model with the predictors
    logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
    logit_reg.fit(x_train, y_train)
    logit_reg_test = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
    logit_reg_test.fit(x_test, y_test)
    # Print coefficients of model
    print('intercept', logit_reg.intercept_[0])
    print(pd.DataFrame({'coeff': logit_reg.coef_[0]}, index=x_train.columns))
    # Print probabilities with decision on test data
    logit_reg_pred = logit_reg.predict(x_test)
    logit_reg_proba = logit_reg.predict_proba(x_test)
    # Print logit results for 0.5 threshold
    logit_result = pd.DataFrame({'actual': y_test,
                                 'p(0)': [p[0] for p in logit_reg_proba],
                                 'p(1)': [p[1] for p in logit_reg_proba],
                                 'predicted': logit_reg_pred})
    print(logit_result)
    # Print logit results for custom threshold
    logit_reg_proba_custom = logit_reg.predict_proba(x_test)[:, 1] # for custom threshold
    logit_reg_pred_custom = (logit_reg_proba_custom >= 0.40).astype(int) # for custom threshold
    logit_result2 = pd.DataFrame({'actual': y_test,
                                  'p(0)': 1 - logit_reg_proba_custom,
                                  'p(1)': logit_reg_proba_custom,
                                  'predicted': logit_reg_pred_custom})
    print(logit_result2)
    
    ## Evaluating model
    # Print confusion matrix
    print('\n--Threshold of 0.5--')
    classificationSummary(y_train, logit_reg.predict(x_train))
    print('')
    classificationSummary(y_test, logit_reg.predict(x_test))
    # Calculate and print metrics
    def log_metrics(y_test, logit_reg_pred):
        tn, fp, fn, tp = confusion_matrix(y_test, logit_reg_pred).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn) # predicted attrition better than guessing
        specificity = tn / (tn + fp) # good for not wasting resources preparing for loss due to attrition
        f1 = 2 * (precision * recall) / (precision + recall)
        print('')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'Specificity: {specificity:.2f}')
        print(f'F1: {f1:.2f}')
    log_metrics(y_test, logit_reg_pred)
    print('\n--Threshold of 0.4--')
    print('Confusion Matrix')
    print(pd.DataFrame(confusion_matrix(y_test, logit_reg_pred_custom)))
    log_metrics(y_test, logit_reg_pred_custom)
    print('')
    
    # ROC curve graph at 0.5 threshold
    fpr, tpr, _ = roc_curve(y_test, logit_reg_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Threshold = 0.5)')
    plt.legend(loc='lower right')
    plt.show()

def run_income(df_income):
    # print heading
    print('*'*60)
    print('*'+' '*58+'*')
    print('*'+' '*13+'Montly Income - Linear Regression'+' '*12+'*')
    print('*'+' '*58+'*')
    print('*'*60)
    # Set predictor and outcome variables
    df_income = get_categories(df_income)
    print(df_income.info())
    predictors = [
                  'JobLevel',
                  'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager', 'JobRole_Manufacturing Director',
                  'JobRole_Research Director', 'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative',
                  ] 
    outcome = 'MonthlyIncome'
    
    x = df_income[predictors]
    y = df_income[outcome]
    # Split data into training and validation sets
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=1)
    # Multiple linear regressio model
    data_lm = LinearRegression()
    data_lm.fit(train_x, train_y)
    # Print model coefficients
    print(pd.DataFrame({'Predictor':x.columns, 'coefficient':data_lm.coef_}))
    # Print model metrics
    print('Training Set')
    regressionSummary(train_y, data_lm.predict(train_x))
    adjr = adjusted_r2_score(train_y, data_lm.predict(train_x), data_lm)
    print(' '*30+f'Adj R^2 : {adjr:.2f}\n')
    # Print model metrics
    print('Validation Set')
    regressionSummary(valid_y, data_lm.predict(valid_x))
    adjr = adjusted_r2_score(valid_y, data_lm.predict(valid_x), data_lm)
    print(' '*30+f'Adj R^2 : {adjr:.2f}\n')

    # Comparing lasso, ridge, and bayesainridge regression models
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    valid_x_scaled = scaler.transform(valid_x)
    # Run each model and print metrics
    print('Lasso')
    lasso = Lasso(alpha=1)
    lasso.fit(train_x_scaled, train_y)
    regressionSummary(valid_y, lasso.predict(valid_x_scaled))
    print('\nRidge')
    ridge = Ridge(alpha=1)
    ridge.fit(train_x_scaled, train_y)
    regressionSummary(valid_y, ridge.predict(valid_x_scaled))
    print('\nBayesianRidge')
    bayesianRidge = BayesianRidge()
    bayesianRidge.fit(train_x_scaled, train_y)
    regressionSummary(valid_y, bayesianRidge.predict(valid_x_scaled))
    
def main():
    # Read in .csv data file
    data = pd.read_csv("ibm.csv")

    ## Exploration and Summary Data
    # print heading
    print('*'*60)
    print('*'+' '*58+'*')
    print('*'+' '*21+'Data Exploration'+' '*21+'*')
    print('*'+' '*58+'*')  
    print('*'*60)
    # Data exploration
    pd.set_option('display.max_columns', None)
    print(data.info())
    print('-'*80+'\n')
    print(data.describe(include='object')) # describe only object columns
    print('-'*80+'\n')
    print(data.describe(include=np.number)) # describe only numeric columns
    print('-'*80+'\n')
    numeric_data = data.select_dtypes(include=['number'])  # select only numeric columns
    plt.figure(figsize=(5, 4))
    sns.heatmap(numeric_data.corr(), cmap='coolwarm')
    plt.show()

    ## Preprocessing
    # Apply LabelEncoder to binary columns
    le = LabelEncoder()
    data['Attrition'] = le.fit_transform(data['Attrition'])  # Yes/No → 1/0
    # Drop columns with no variance
    data.drop(columns=['StandardHours', 'Over18', 'EmployeeCount'], inplace=True)
    # Drop uneeded columns (either poorly defined or highly correlated with another variable)
    data.drop(columns=['DailyRate', 'HourlyRate', 'MonthlyRate', 'EmployeeNumber', 'PerformanceRating'], inplace=True)

    # Copy dataset into two separate dataframes to work independently
    df_attrition = data.copy()
    df_income = data.copy()

    run_attrition(df_attrition)
    run_income(df_income)
    
if __name__ == "__main__":
    main()
