import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

print("Loading data...")
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
print("Initial shape:", df.shape)

# Drop missing values
df = df.dropna()
print("Shape after dropna:", df.shape)

# Rename columns
df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'})

# Filter location
df = df[df.Location.isin(['Melbourne', 'MelbourneAirport', 'Watsonia'])]
print("Shape after location filter:", df.shape)

# Season mapping
def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'

df['Date'] = pd.to_datetime(df['Date'])
df['Season'] = df['Date'].apply(date_to_season)
df = df.drop(columns='Date')

print("Columns in df:", df.columns.tolist())

# X and y
X = df.drop(columns='RainToday', axis=1)
y = df['RainToday']

print("Class balance (y.value_counts()):")
print(y.value_counts())
print("Class percentages:")
print(y.value_counts(normalize=True))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)

# Detect numeric and categorical
numeric_features = X_train.select_dtypes(include=['float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Random Forest Pipeline
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid_rf = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Running GridSearchCV on Random Forest ---")
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

print("RF Best params:", grid_search_rf.best_params_)
print("RF Best CV score:", grid_search_rf.best_score_)
test_score_rf = grid_search_rf.score(X_test, y_test)
print(f"RF Test set score: {test_score_rf:.4f}")

y_pred_rf = grid_search_rf.predict(X_test)
print("\nRF Classification Report:")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf, labels=['No', 'Yes'])
print("RF Confusion Matrix (labels=['No', 'Yes']):")
print(cm_rf)
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
tpr_rf = tp_rf / (tp_rf + fn_rf)
print(f"RF TN: {tn_rf}, FP: {fp_rf}, FN: {fn_rf}, TP: {tp_rf}")
print(f"RF True Positive Rate (TPR / Recall for 'Yes'): {tpr_rf:.4f} ({tpr_rf*100:.2f}%)")

# Feature importances
feature_names = numeric_features + list(grid_search_rf.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

feature_importances_rf = grid_search_rf.best_estimator_['classifier'].feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances_rf}).sort_values(by='Importance', ascending=False)
print("\nTop 10 Features (Random Forest):")
print(importance_df.head(10))

# Logistic Regression
print("\n--- Running Logistic Regression ---")
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

param_grid_lr = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_lr.fit(X_train, y_train)

print("LR Best params:", grid_search_lr.best_params_)
print("LR Best CV score:", grid_search_lr.best_score_)
test_score_lr = grid_search_lr.score(X_test, y_test)
print(f"LR Test set score: {test_score_lr:.4f}")

y_pred_lr = grid_search_lr.predict(X_test)
print("\nLR Classification Report:")
print(classification_report(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr, labels=['No', 'Yes'])
print("LR Confusion Matrix (labels=['No', 'Yes']):")
print(cm_lr)
tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()
tpr_lr = tp_lr / (tp_lr + fn_lr)
print(f"LR TN: {tn_lr}, FP: {fp_lr}, FN: {fn_lr}, TP: {tp_lr}")
print(f"LR True Positive Rate (TPR / Recall for 'Yes'): {tpr_lr:.4f} ({tpr_lr*100:.2f}%)")
