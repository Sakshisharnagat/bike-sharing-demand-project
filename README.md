"Machine Learning project to predict bike rental demand using regression models like Random Forest and Gradient Boosting."
Project Overview
In this project, I worked on predicting daily bike rental demand using machine learning models. The idea was to understand how different algorithms perform on real-world data and how accurately they can estimate the number of bikes rented on a given day.
The dataset used is day.csv, which contains daily records including weather conditions, season, temperature, and total rental count. The main goal is to predict the cnt value using these features.

Algorithms Implemented

1. Linear Regression
2.Decision Tree Regressor
3.Random Forest Regressor
4. K-Nearest Neighbors (KNN)
5. Support Vector Regression (SVR)
6. Gradient Boosting Regressor

Key Results (80:20 Split)
MODE	MAE	MSE	RMSE	R2 Score
Linear Regression	617.3930	6.9103	831.2851	0.8276
Decision Tree	614.89	7.4291	881.9374
	0.8060
Random Forest	443.6259	4.6879	677.0871	0.8856
KNN	571.8231	6.7918	824.1280	0.8307
Gradient Boosting	464.1140	4.3622	662.321	0.8902
SVR	1689.9563	3.9847	1996.1895	0.0062

•	Best Performing Model: Gradient Boosting Regressor (highest R² score with lower error values)
Implementation Steps
1. Setup
pip install pandas scikit-learn numpy matplotlib seaborn

2. Load & Preprocess Data
import pandas as pd

df = pd.read_csv('day.csv')

categorical_cols = ['season', 'weathersit']

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
3. Train Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = df_encoded.drop('cnt', axis=1)
y = df_encoded['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"R2 Score: {r2_score(y_test, predictions):.3f}")
4. Train Other Models (Same Pattern)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

models = {
'Decision Tree': DecisionTreeRegressor(),
'Random Forest': RandomForestRegressor(),
'KNN': KNeighborsRegressor(),
'SVR': SVR(),
'Gradient Boosting': GradientBoostingRegressor()
}

for name, model in models.items():
model.fit(X_train, y_train)

preds = model.predict(X_test)

r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"{name}")
print(f"R2: {r2:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
print("------")

Project Structure
bike-sharing-demand-project/
│
├── dataset/
│   └── day.csv
│
├── notebooks/
│   ├── ML_model1_linear.ipynb
│   ├── ML_model2_decision.ipynb
│   ├── ML_model3_random.ipynb
│   ├── ML_model4_knn.ipynb
│   ├── ML_model5_svr.ipynb
│   ├── ML_model6_gradient.ipynb
│   └── model_performance.ipynb
│
└── README.md


Methodology
The project starts with loading the dataset from a GitHub-hosted source. After loading the data, preprocessing is performed which includes handling missing values, encoding categorical features, and applying scaling where required. Once the data is ready, multiple regression models are trained and tested on the dataset. 
For evaluating model performance, metrics such as R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are used. Each model is evaluated using two different train-test splits: 80:20 (Training:Testing) and 70:30 (Training:Testing). This helps in comparing how models perform under different data distributions.
Repository Structure
Individual model implementations are organized into separate notebooks for better clarity. A final combined notebook is used to compare all models and analyze their performance together. The dataset (day.csv) is stored within the repository and accessed directly during execution.
Tools and Technologies
The project is developed using Python, with libraries such as Pandas and NumPy for data handling, Scikit-learn for model building, and Matplotlib and Seaborn for visualization. The entire implementation is carried out using Google Colab, and GitHub is used for storing the code, dataset, and maintaining version control.
Collaboration
This project is developed as part of Machine Learning TAE-1, where the work mainly focuses on implementing multiple regression models, evaluating their performance, and comparing results in a structured manner.
Vaishnavi Sandokar – Linear Regression, Decision Tree, Random Forest Regression.Sakshi Sharnagat– SVR, Gradient Boosting Regression, KNN Regression.
