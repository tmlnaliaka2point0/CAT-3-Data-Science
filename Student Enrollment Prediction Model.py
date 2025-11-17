import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Reproducibility Seed
np.random.seed(42)

# 1. GENERATED SYNTHETIC DATASET 
print(" 1. Generating Synthetic Student Data ")
n_students = 2000 # Increased size for better simulation

# Create the Features (X)
X_data = pd.DataFrame({
    # Numerical Features (Academic/Financial)
    'High_School_GPA': np.random.normal(3.4, 0.5, n_students).clip(1.5, 4.0),
    'SAT_Score': np.random.randint(900, 1600, n_students),
    'Financial_Aid_Amount': np.random.randint(0, 30000, n_students),
    'Application_Fee_Waived': np.random.choice([0, 1], n_students, p=[0.7, 0.3]),

    # Categorical Features (Demographic/Program)
    'Program_Choice': np.random.choice(['Engineering', 'Arts', 'Business', 'Science', 'Health'], n_students, p=[0.25, 0.15, 0.25, 0.2, 0.15]),
    'First_Gen': np.random.choice([0, 1], n_students, p=[0.65, 0.35]), # 1 if first-generation
    'Distance_Region': np.random.choice(['Local', 'Regional', 'Out_of_State'], n_students, p=[0.5, 0.3, 0.2]),
})

# Create the Target Variable (y): Enrollment (1 = Yes, 0 = No)
# The probability is a function of high GPA, high SAT, and high financial aid.
prob_enroll = (
    0.3 +                                                                        # Base probability
    (X_data['High_School_GPA'] / 4.0) * 0.2 +                                    # GPA impact
    (X_data['SAT_Score'] / 1600.0) * 0.15 +                                      # SAT impact
    (X_data['Financial_Aid_Amount'] / 30000.0) * 0.1 -                            # Financial Aid impact
    (X_data['First_Gen'] * 0.05) +                                               # First-Gen slight negative bias (higher support needed)
    (X_data['Distance_Region'].apply(lambda x: 0.0 if x == 'Local' else -0.1))   # Distance penalty
)
prob_enroll = prob_enroll.clip(0.1, 0.9) # Clip probabilities to be realistic

y_data = (np.random.rand(n_students) < prob_enroll).astype(int)

# Combine into a single DataFrame (for clarity, though X/y split is necessary)
df_students = X_data.copy()
df_students['Enrolled'] = y_data

print(f"Total Records Generated: {len(df_students)}")
print(f"Overall Enrollment Rate: {df_students['Enrolled'].mean():.2f}")
print("\n Sample of Generated Data ")
print(df_students.head())

# 2. DATA SPLIT (Generating Train/Test Datasets) 
print("\n 2. Splitting Data into Training and Testing Sets ")
TARGET = 'Enrolled'
X = df_students.drop(TARGET, axis=1)
y = df_students[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Create the final training and testing DataFrames, including the target variable
df_train = X_train.assign(Enrolled=y_train)
df_test = X_test.assign(Enrolled=y_test)

print(f"\nTraining Dataset (df_train) created with {len(df_train)} records.")
print(f"Testing Dataset (df_test) created with {len(df_test)} records.")

# 3. MODELING PIPELINE 
print("\n 3. Training and Evaluating Model")

# Define feature types for preprocessing
numerical_features = ['High_School_GPA', 'SAT_Score', 'Financial_Aid_Amount']
binary_features = ['Application_Fee_Waived', 'First_Gen'] # Treat binary features as categorical
categorical_features = ['Program_Choice', 'Distance_Region']

# Create Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features + binary_features)
    ],
    remainder='passthrough'
)

# Defining model (Random Forest)
model_algorithm = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')

# Full modeling pipeline
enrollment_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model_algorithm)
])

# Train the model
enrollment_model_pipeline.fit(X_train, y_train)

# 4. MODEL EVALUATION 
y_pred = enrollment_model_pipeline.predict(X_test)
y_pred_proba = enrollment_model_pipeline.predict_proba(X_test)[:, 1]

# Display Key Metrics
print("\n Model Evaluation on Test Set ")
print(f"Area Under the ROC Curve (AUC): {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Actionable Insight: Feature Importance
feature_names = numerical_features + list(
    enrollment_model_pipeline.named_steps['preprocessor']
    .named_transformers_['cat']
    .get_feature_names_out(categorical_features + binary_features)
)

#IMPORTANCES
importances = enrollment_model_pipeline.named_steps['classifier'].feature_importances_
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\n Top 5 Actionable Feature Importances")
print(feature_importances.head(5).to_string())

# Plotting Feature Importance
plt.figure(figsize=(10, 6))
feature_importances.head(10).plot(kind='barh')
plt.title('Top 10 Feature Importances for Enrollment Prediction')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.show()
