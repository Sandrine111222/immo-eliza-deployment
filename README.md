# Description 📊

The project analyzes Belgian real estate data sourced from **ImmoVlan**.

This subpart focuses on deployment and creations of apps. Web developers should be able access the predictions whenever they need to. They also will have access to a small web application for the non-technical employees and possibly their clients to use. 
The goal is to be able to predict prices of a house with desired features using these user-friendly apps. 


---

# Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Creating FastAPI and Docker](#creating-FastAPI-and-Docker)  
- [Creating Streamlit](#creating-streamlit)  
- [Make the work publicly accessible](#evaluate-the-performance-of-a-model)   
- [Project Structure](#project-structure)   


---

# Project Overview

The project includes:

- Preprocessing data for machine learning  
- Applying linear regression in a real-life context  
- Exploring multiple regression models  
- Evaluating model performance  
- Using hyperparameter tuning and cross-validation  

---

# Features

- Handling missing values using imputation  
- Converting categorical variables using one-hot encoding  
- Standardizing numerical features  
- Tracking preprocessing steps with a pipeline  
- Splitting the dataset for training/evaluation  
- Ensuring a reusable and clean ML workflow  

---

# Dataset

The cleaned dataset contains:

- **15,000+ property listings**  
- **17 features**, including:  
  - Living area  
  - Build year  
  - Number of rooms  
  - Number of facades  
  - Property type  
  - Province & region  
  
---

# Creating Pipeline

### Key steps

1. **Clean data**  
   - Handle duplicates  
   - Manage missing values  
   - Drop unnecessary rows/columns  

2. **Preprocess data**  
   - Impute missing values  
   - Encode categorical features  
   - Scale numerical features  

3. **Train model**  
   - Split dataset  
   - Fit model  

4. **Predict**

5. **Evaluate model**  
   - R², MSE, MAE  
   - Detect overfitting  

---

# Explore Machine Learning Models

- Baseline: **Linear Regression**  
- Compare with non-linear models such as:  
  - Decision Tree  
  - Random Forest  
  - Support Vector Regression  
  - XGBoost  

---

# Evaluate the Performance of a Model

| Model                    | RMSE        | MAE         | R²     |
| ------------------------ | ----------- | ----------- | ------ |
| **Linear Regression**    | 176,555.453 | 103,023.992 | 0.511  |
| **Decision Tree**        | 181,462.994 | 98,865.731  | 0.483  |
| **Random Forest**        | 147,918.026 | 73,414.115  | 0.657  |
| **XGBoost**              | 146,005.958 | 76,646.448  | 0.665  |

Overfitting might be occurring (R² train values are too high compared to R² test).

---


# Project Structure

```bash
IMMOELIZA_ML/
│
├── Cleaning
│
├── Predict.py
│
├── test.ipynb
|
|── house.py
|
├── test.py
│
└── README.md


---


This project is part of AI & Data Science Bootcamp training at **`</becode>`** and it written by :

- Sandrine Herbelet  [LinkedIn](https://www.linkedin.com/in/) | [Github](https://github.com/Sandrine111222)

