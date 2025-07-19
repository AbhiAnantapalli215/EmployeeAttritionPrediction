import numpy as np                                                              # type:ignore
import scipy as sp                                                              # type:ignore
import pandas as pd                                                             # type:ignore
from flask import Flask,request,render_template                                 # type:ignore
import pickle                                                                   # type:ignore

app=Flask(__name__)
with open('model.pkl','rb') as f:
    model = pickle.load(f)
with open('encoder.pkl','rb') as f:
    encoder = pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        Age=request.form.get("Age")
        BusinessTravel=request.form['BusinessTravel']
        Department=request.form['Department']
        DistanceFromHome=request.form.get("DistanceFromHome")
        EducationField=request.form['EducationField']
        EnvironmentSatisfaction=request.form.get("EnvironmentSatisfaction")
        JobLevel=request.form.get("JobLevel")
        JobRole=request.form['JobRole']
        JobSatisfaction=request.form.get("JobSatisfaction")
        MaritalStatus=request.form['MaritalStatus']
        MonthlyIncome=request.form.get("MonthlyIncome")
        NumCompaniesWorked=request.form.get("NumCompaniesWorked")
        OverTime=request.form['OverTime']
        RelationshipSatisfaction=request.form.get("RelationshipSatisfaction")
        StockOptionLevel=request.form.get("StockOptionLevel")
        TotalWorkingYears=request.form.get("TotalWorkingYears")
        WorkLifeBalance=request.form.get("WorkLifeBalance")
        YearsAtCompany=request.form.get("YearsAtCompany")
        YearsSinceLastPromotion=request.form.get("YearsSinceLastPromotion")
        YearsWithCurrManager=request.form.get("YearsWithCurrManager")

        dict={
            'Age':int(Age),
            'BusinessTravel':str(BusinessTravel),
            'Department':Department,
            'DistanceFromHome':int(DistanceFromHome),
            'EducationField':str(EducationField),
            'EnvironmentSatisfaction':int(EnvironmentSatisfaction),
            'JobLevel':int(JobLevel),
            'JobRole':JobRole,
            'JobSatisfaction':int(JobSatisfaction),
            'MaritalStatus':str(MaritalStatus),
            'MonthlyIncome':int(MonthlyIncome),
            'NumCompaniesWorked':int(NumCompaniesWorked),
            'OverTime':str(OverTime),
            'RelationshipSatisfaction':int(RelationshipSatisfaction),
            'StockOptionLevel':StockOptionLevel,
            'TotalWorkingYears':int(TotalWorkingYears),
            'WorkLifeBalance':int(WorkLifeBalance),
            'YearsAtCompany':int(YearsAtCompany),
            'YearsSinceLastPromotion':int(YearsSinceLastPromotion),
            'YearsWithCurrManager':int(YearsWithCurrManager)
        }

        df=pd.DataFrame([dict])

        #Encoding Categorical features
        categorical_features=['BusinessTravel','Department','EducationField','JobRole','MaritalStatus','OverTime']
        for feature in categorical_features:
            if feature in encoder:
                df[feature]=encoder[feature].transform(df[feature])

        #Scaling Numerical features
        numerical_features=['Age','DistanceFromHome','EnvironmentSatisfaction','JobLevel',
                            'JobSatisfaction','MonthlyIncome','NumCompaniesWorked',
                            'RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
                            'WorkLifeBalance','YearsAtCompany','YearsSinceLastPromotion',
                            'YearsWithCurrManager']
        df[numerical_features]=scaler.transform(df[numerical_features])

        #Making prediction
        prediction=model.predict(df)
        #print(prediction)
        if prediction=='No':
            return render_template('result.html',prediction_text='Employee Might Not Leave The Job')
        else:
            return render_template('result.html',prediction_text='Employee Might Leave The Job')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)