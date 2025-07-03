
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("C:/Users/Likithasri/Downloads/Loan payments data.csv")
df.dropna(inplace=True)

# 2. Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 3. Define features and label
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 5. Train Decision Tree with GridSearch
param_grid = {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
clf.fit(X_train, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test)
print("Best Params:", clf.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf.best_estimator_, feature_names=X.columns, class_names=["Not Approved", "Approved"], filled=True)
plt.savefig("loan_tree.png")
plt.show()

# 8. Save model
with open("loan_model.pkl", "wb") as f:
    pickle.dump(clf.best_estimator_, f)