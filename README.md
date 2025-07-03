LOAN APPROVAL PREDICTION :  /// Problem Statement   :    The goal of my project was to build a machine learning model that can predict whether a loan application will be approved based on historical data. I worked with a real-world loan payments dataset, cleaned and encoded the features, and trained a Decision Tree classifier tuned with GridSearchCV for optimal performance. This model could help financial institutions automate approvals, reduce manual effort, and make faster decisions while maintaining fairness and accuracy.”
### 🛠 Technologies Used

- **Python**
- **pandas**, **NumPy**
- **scikit-learn**
- **matplotlib**
- **pickle**

---

### 📊 Dataset Details

- **Source**: Kaggle Loan Payments Data
- **File Name**: `Loan payments data.csv`
- **Target Column**: `loan_status`
- Categorical columns are label-encoded
- Missing values are dropped for cleaner training

---

### 🧪 Project Workflow

1. Load and clean the dataset (`dropna`)
2. Encode categorical variables using `LabelEncoder`
3. Define features (`X`) and target (`y`)
4. Split data with `train_test_split` (stratified)
5. Perform hyperparameter tuning with `GridSearchCV`
6. Train a `DecisionTreeClassifier`
7. Evaluate accuracy, confusion matrix, and classification report
8. Visualize the decision tree and save it as `loan_tree.png`
9. Export the best model as `loan_model.pkl`

---

### 🚀 How to Run This Project (Step-by-Step)

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Likitha-sd/LoanApprovalPredictionProject.git
