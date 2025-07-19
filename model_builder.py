import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("datasets/data.csv")

# Target and features
x = data.drop(columns=["Attrition"])
y = data["Attrition"]

# Selected features
selected_features = [
    'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'EducationField',
    'EnvironmentSatisfaction', 'JobLevel', 'JobRole', 'JobSatisfaction', 
    'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
    'WorkLifeBalance', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]
x_selected = x[selected_features].copy()

# Encode categorical features
encoder = {}
categorical_features = x_selected.select_dtypes(include=[object]).columns
for feature in categorical_features:
    le = LabelEncoder()
    x_selected[feature] = le.fit_transform(x_selected[feature])
    encoder[feature] = le

# Save encoders
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Scale numerical features
numerical_features = [
    'Age','DistanceFromHome','EnvironmentSatisfaction','JobLevel',
    'JobSatisfaction','MonthlyIncome','NumCompaniesWorked',
    'RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
    'WorkLifeBalance','YearsAtCompany','YearsSinceLastPromotion',
    'YearsWithCurrManager'
]
scaler = StandardScaler()
x_selected[numerical_features] = scaler.fit_transform(x_selected[numerical_features])

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'lr': LogisticRegression(C=0.5, max_iter=200, random_state=42),
    'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'xgb': XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=5, random_state=42, eval_metric='logloss'),
    'knn': KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'svm': SVC(C=1.5, probability=True, random_state=42),
    'gb': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
}

# Voting ensemble
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='soft',
    weights=[0.12, 0.2, 0.3, 0.08, 0.15, 0.15]
)

# Train model
ensemble.fit(x_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(ensemble, f)

print("✅ Model, encoder, and scaler saved successfully.")
