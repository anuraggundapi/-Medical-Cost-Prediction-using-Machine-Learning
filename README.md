# -Medical-Cost-Prediction-using-Machine-Learning

# 🧠 Medical Cost Prediction using Machine Learning

This project predicts the **individual medical insurance cost** using ML algorithms based on demographic and health-related features like age, BMI, smoking status, etc.

---

## 📝 Project Objective

This Data Science project aims to predict individual medical insurance expenses using a combination of demographic and lifestyle features. The goal is to assist insurance companies in estimating premium charges more accurately.
To help insurance companies estimate the **medical expenses of a person** by analyzing the relationship between various features such as:
- Age
- BMI
- Number of Children
- Smoking Status
- Region

This can aid in **premium decision-making** and customer profiling.

---

## 📂 Dataset

- Source: `insurance.csv` (1338 records, 7 features)
- Features:
            age: Age of primary beneficiary
            sex: Male or Female
            bmi: Body Mass Index
            children: Number of dependents
            smoker: Yes/No
            region: Location in US (NE, SE, SW, NW)
            expenses: Final insurance cost

---

🔎 Exploratory Data Analysis (EDA)
Checked for nulls, duplicates
Categorical vs Numerical column separation
Heatmap for correlation
Pairplot to study feature relationships
Created age bins, smoker/gender expense charts
Identified and handled outliers in bmi using IQR method

💡 Feature Engineering
Created bmi_cap to clip extreme BMI values
Encoded categorical variables (sex, smoker, region) using pd.get_dummies()
Created input features:
age, bmi_cap, children, smoker_yes, region_southeast
---

🧪 Model Building & Selection
Models Trained:
Multiple Linear Regression
Lasso Regression
Ridge Regression
Random Forest Regressor
XGBoost Regressor ✅ (Best)

📈 Evaluation Metrics Used:
MAE (Mean Absolute Error): Average error magnitude
MSE (Mean Squared Error): Penalizes large errors
RMSE (Root Mean Squared Error): Interpretable (same unit)
R² Score: Explains variance, closer to 1 is better

| Model              | MAE     | RMSE    | R² Score |
| ------------------ | ------- | ------- | -------- |
| Linear Regression  | 3850.09 | 7150.27 | 0.72     |
| Lasso Regression   | 5142.60 | 9222.12 | 0.54     |
| Ridge Regression   | 3849.68 | 7147.69 | 0.72     |
| XGBoost (Selected) | 2088.04 | 4416.36 | **0.89** |
| Random Forest      | 2328.24 | 4685.04 | 0.88     |

---

## 📊 Evaluation Metrics

- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Goodness-of-fit)

---

## 🧠 Deployment (Streamlit App)

Built an interactive Streamlit app (`app.py`) that:
- Takes user input (age, bmi, children, smoker, region)
- Predicts medical expenses
- Displays estimated cost with explanation

### 👉 To Run:
```bash
streamlit run app/app.py

🛠 Tools & Libraries
Python
Pandas, NumPy, Matplotlib, Seaborn
Scikit-learn, XGBoost
Statsmodels
Streamlit
Pickle
