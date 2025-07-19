import pandas as pd                                                                 # type: ignore
import pickle                                                                       # type: ignore
from sklearn.model_selection import train_test_split                                # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder                      # type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier     # type: ignore
from sklearn.linear_model import LogisticRegression                                 # type: ignore
from xgboost import XGBClassifier                                                   # type: ignore
from sklearn.neighbors import KNeighborsClassifier                                  # type: ignore
from sklearn.svm import SVC                                                         # type: ignore 
from sklearn.ensemble import VotingClassifier                                       # type: ignore
from sklearn.metrics import accuracy_score                                          # type: ignore

#Loading data into a DataFrame
data=pd.read_csv('datasets/data.csv')
x=data.drop(columns=['Attrition'])
y=data['Attrition']


#Step 1: Feature Selection (Selecting 20 best features after analyzing with the help of SelectKBest())
selected_features=[
    'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'EducationField',
    'EnvironmentSatisfaction', 'JobLevel', 'JobRole', 'JobSatisfaction', 
    'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
    'WorkLifeBalance', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]
x_selected=x[selected_features].copy()

#Step 2: Encoding Categorical Features
encoder={}
cf=x_selected.select_dtypes(include=[object]).columns
for feature in cf:
    l=LabelEncoder()
    x_selected[feature] = l.fit_transform(x_selected[feature])
    encoder[feature]=l
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

#Step 3: Scaling Numerical Features
nf=['Age','DistanceFromHome','EnvironmentSatisfaction','JobLevel',
    'JobSatisfaction','MonthlyIncome','NumCompaniesWorked',
    'RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
    'WorkLifeBalance','YearsAtCompany','YearsSinceLastPromotion',
    'YearsWithCurrManager']
scaler=StandardScaler()
x_selected[nf] = scaler.fit_transform(x_selected[nf])
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


#Step 4: New DataFrame with the selected features
x_train, x_test, y_train, y_test=train_test_split(x_selected, y, test_size=0.2, random_state=42)


#Step 5: Picking different ML models
logistic_reg=LogisticRegression(C=0.5, max_iter=200, random_state=42)
random_forest=RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
xgboost=XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=5, random_state=42, eval_metric='logloss')
knn=KNeighborsClassifier(n_neighbors=7, weights='distance')
svm=SVC(C=1.5, probability=True, random_state=42)
gradient_boost=GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)


#Step 6: Assigning Weights for Different Models
ensemble_model=VotingClassifier(
    estimators=[
        ('lr', logistic_reg),
        ('rf', random_forest),
        ('xgb', xgboost),
        ('knn', knn),
        ('svm', svm),
        ('gb', gradient_boost)
    ],
    voting='soft',
    weights=[0.12, 0.2, 0.3, 0.08, 0.15, 0.15]
)

#Step 7: Fitting final model
ensemble_model.fit(x_train, y_train)
#y_pred=ensemble_model.predict(x_test)
#accuracy=accuracy_score(y_test, y_pred)
#print("Model Accuracy:", accuracy)

#Step 8: Saving the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)
model=pickle.load(open('model.pkl','rb'))