<html>
  <body>
    <a href="https://colab.research.google.com/drive/1s8hRDZQ71k_WMWmr99lVKWUoq-EMBpQl#scrollTo=9MexvIQyAW_y">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
    
# Zindi utility fraud detection  
## 💡 Capstone Project: Big Data Analytics
---

## 📘 Project Overview
This project was developed as part of the **Zindi Fraud Detection Challenge**. The goal is to build a machine learning classification model capable of predicting the likelihood of customers committing fraud based on their electricity and gas consumption patterns.

The challenge is hosted by **Zindi** in collaboration with **The Tunisian Company of Electricity and Gas (STEG)**, which has experienced significant losses due to fraudulent activities from customers. The solution leverages historical client and invoice data to train a predictive model that can help detect future fraud risks proactively.

---
## 🎯 Problem Statement

> **How can STEG detect fraudulent activities from their customers while improving satisfaction and maintaining operational efficiency?**

The objective is to classify customers into two categories:
- **Fraudulent (1)**
- **Not Fraudulent (0)**

Using past behavior and billing data, we aim to identify suspicious patterns using supervised machine learning models.

---

## 📊 Dataset Information and 🏷️ Sector of Focus

**Sector:** Cybersecurity  
**Problem Statement:** (Already Explained) <br>
**Dataset Title:** Fraud Detection in Electricity and Gas Consumption Challenge  <br>
**Source Link:** https://zindi.africa/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge  <br>
**Number of Rows and Columns:**  test_df ===> (1939730, 20)  train_df ===> (4476749, 21)  <br>
**Data Structure:** Structured (CSV) <br>
*Unstructured (Text, Images):* N/A <br>
**Data Status:** Requires Preprocessing <br>

---

## 🧠 Python Analytics Tasks

The Python notebook includes the following components:

### 1. 🔧 Data Cleaning
- Loading data using panda analysing dataset information
```python
import pandas as pd
train_client= pd.read_csv('drive/MyDrive/train/client_train.csv', low_memory=False)
train_invoice= pd.read_csv('drive/MyDrive/train/invoice_train.csv', low_memory=False)
test_invoice= pd.read_csv('drive/MyDrive/test/invoice_test.csv', low_memory=False)
test_client=pd.read_csv('drive/MyDrive/test/client_test.csv', low_memory=False)
#First merge the dataset to have a common target client_id
train_df = pd.merge(train_invoice, train_client, on='client_id')
test_df = pd.merge(test_invoice, test_client, on='client_id')
#check column data types
train_df.dtypes
test_df.dtypes
train_df.isnull()
test_df.isnull()
# train_df.isnull().sum()
# test_df.isnull().sum()
# Data set Structure
train_df.head()
# Data set Structure
test_df.head(10)
train_df.describe()
test_df.describe()
# Check the shape (rows, columns)
test_df.shape
train_df.shape
```
- Missing value imputation
```python
# 1. Clean the Dataset
# ▪ Handle missing values, inconsistent formats, and outliers
# ▪ Apply necessary data transformations (e.g., encoding, scaling)

import pandas as pd
from typing import List

def fill_numerical_missing(train_df: pd.DataFrame, test_df: pd.DataFrame, columns: List[str] = None) -> None:
    """
    Fill missing values in numerical columns with the mean of train_df.
    """
    if columns is None:
        columns = train_df.select_dtypes(include='number').columns.tolist()
    for col in columns:
        mean_val = train_df[col].mean()
        train_df[col].fillna(mean_val, inplace=True)
        if col in test_df.columns:
            test_df[col].fillna(mean_val, inplace=True)

def fill_categorical_missing(train_df: pd.DataFrame, test_df: pd.DataFrame, columns: List[str] = None) -> None:
    """
    Fill missing values in categorical columns with the mode of train_df.
    """
    if columns is None:
        columns = train_df.select_dtypes(include='object').columns.tolist()
    for col in columns:
        if train_df[col].isnull().any():
            mode_val = train_df[col].mode()[0]
            train_df[col].fillna(mode_val, inplace=True)
            if col in test_df.columns:
                test_df[col].fillna(mode_val, inplace=True)
```
- Format standardization  
- Outlier detection and removal  
- Encoding & scaling

### 2. 📊 Exploratory Data Analysis (EDA)
- Descriptive statistics
  ```python

  ```
- Correlation heatmaps
 ```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_correlation_heatmap(
    df,
    title="Correlation Heatmap",
    max_features=None,
    annot=True,
    fmt=".2f",
    figsize=(12, 8),
    cmap="coolwarm",
    annot_fontsize=10,
    xtick_fontsize=12,
    ytick_fontsize=12,
    mask_upper=True,
    min_corr_display=0.1
):
    """
    Plots an improved, readable correlation heatmap, with options for column reduction and visibility.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        title (str): Title for the plot.
        max_features (int, optional): Show only top N features most correlated with target.
        annot (bool): Whether to annotate cells.
        fmt (str): Annotation format.
        figsize (tuple): Figure size.
        cmap (str): Color map.
        annot_fontsize (int): Font size for annotations.
        xtick_fontsize (int): Font size for x-tick labels.
        ytick_fontsize (int): Font size for y-tick labels.
        mask_upper (bool): Show only lower triangle if True.
        min_corr_display (float): Only show correlations above this absolute value.
    """
    corr = df.select_dtypes(include='number').corr()
    
    # Reduce columns if max_features is set
    if max_features is not None and "target" in corr.columns:
        top_features = corr["target"].abs().sort_values(ascending=False).head(max_features+1).index
        corr = corr.loc[top_features, top_features]

    # Mask upper triangle for clarity
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    # Only show significant correlations
    display_corr = corr.copy()
    display_corr[(abs(display_corr) < min_corr_display) & (display_corr != 1.0)] = np.nan

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        display_corr,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        mask=mask,
        annot_kws={"size": annot_fontsize},
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=xtick_fontsize, rotation=45, ha='right')
    plt.yticks(fontsize=ytick_fontsize, rotation=0)
    plt.tight_layout()
    plt.show()
   # from plotting_utils import encode_target_column, plot_correlation_heatmap

   # Encode target column if needed
   train_df = encode_target_column(train_df, 'target')

   # Plot improved correlation heatmap, showing only top 10 features
   plot_correlation_heatmap(train_df, max_features=10)
 ```
- Distribution plots
 ```python
   import seaborn as sns
import matplotlib.pyplot as plt

# Histogram of target
sns.countplot(data=train_df, x='target')
plt.title("Distribution of Target Variable")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()

# Distribution of numeric feature (example: 'amount')
if 'amount' in train_df.columns:
    sns.histplot(data=train_df, x='amount', bins=30, kde=True)
    plt.title("Distribution of Amount")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.show()
 ```

```python
#Visualize fraudulent activities
fraudactivities = train_client.groupby(['target'])['client_id'].count()
plt.bar(x=fraudactivities.index, height=fraudactivities.values, tick_label = [0,1])
plt.title('Fraud - Target Distribution')
plt.show()
```
- Relationship visualizations
  ```python
  # from plotting_utils import encode_target_column, plot_correlation_heatmap

  # Encode target column if needed
  train_df = encode_target_column(train_df, 'target')
  
  # Plot improved correlation heatmap, showing only top 10 features
  plot_correlation_heatmap(train_df, max_features=10)
  # from plotting_utils import plot_count_by_category # This line caused the error

  plot_count_by_category(train_df, 'client_catg', rotate_xticks=45) # Changed 'client_type' to 'client_catg' based on the available columns
  
  ```

### 3. 🤖 Modeling
- Selected Model: **Random Forest**, KMeans, Linear Regression
  ```python
   from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score, classification_report
  
  # Train model
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)
  
  # Predict
  y_pred = model.predict(X_val)
  
  # Evaluate
  print("Accuracy:", accuracy_score(y_val, y_pred))
  print("Classification Report:\n", classification_report(y_val, y_pred))
  ```
- Modeling approach: Classification / Regression / Clustering
  ```python
  # Apply a Machine Learning or Clustering Model====>>>>>I used classification
  # Drop unnecessary columns (like IDs, if not useful for prediction)
  X = train_df.drop(columns=['target', 'invoice_id', 'client_id'], errors='ignore')
  y = train_df['target']
  # Encode and Scale Features
  from sklearn.preprocessing import LabelEncoder, StandardScaler
  
  # Label Encoding for categorical columns
  for col in X.select_dtypes(include='object').columns:
      le = LabelEncoder()
      X[col] = le.fit_transform(X[col])
  
  # Scale numeric features
  scaler = StandardScaler()
  X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
  ```
- Training and testing dataset splits
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
  
  ```

### 4. 📈 Model Evaluation
- Evaluation Metrics Used: Accuracy / Precision / Recall / RMSE / Silhouette Score
  ```python
    from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score, classification_report
  
  # Train model
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)
  
  # Predict
  y_pred = model.predict(X_val)
  
  # Evaluate
  print("Accuracy:", accuracy_score(y_val, y_pred))
  print("Classification Report:\n", classification_report(y_val, y_pred))
  ```
- Confusion Matrix or Summary Table

### 5. 🛠️ Code Structure
- Modularized functions and scripts  
- Rich markdown explanations and inline comments  
- Easy to follow and reproducible

### 6. 🌟 Innovation
- [Describe any creative or unique approach you implemented, e.g., ensemble modeling, feature engineering, etc.]

---

## 📌 Power BI Dashboard

An interactive dashboard was developed to visualize the analytical results. Key features include:

- 📌 **Overview Page**: Project context and summary insights  
- 📈 **Visuals**: Bar charts, pie charts, line graphs, scatter plots  
- 🎚 **Filters & Slicers**: Date ranges, categories, dynamic comparisons  
- 🧠 **Advanced Features**: DAX formulas, custom tooltips, bookmarks  
- 🧪 **Insights**: Actionable recommendations derived from the data

> 🖼️ **Screenshots** of the dashboard are available in the `/screenshots` folder.

---
## Recommendations: Data-Driven Suggestions

### Prioritize High-Risk Accounts for Inspection
Use model risk scores to focus field audits and investigations on accounts most likely to commit fraud, optimizing resource allocation.

### Implement Real-Time Consumption Monitoring
Deploy automated systems to flag and alert unusual or suspicious usage patterns as they happen, enabling quick intervention.

### Enhance Customer Verification Processes
Strengthen onboarding and periodic re-verification for customer segments or meter types with elevated fraud risk.

### Develop Fraud Awareness Campaigns
Launch educational initiatives targeting high-risk groups to inform them about fraud consequences and promote honest consumption reporting.

### Continuously Update the Detection Model
Regularly retrain your fraud detection model with new data and confirmed fraud cases to adapt to evolving fraud tactics and maintain high accuracy.

---

## 🗂️ Repository Structure

```bash
📁 capstone-big-data-project/
├── data/
│   └── dataset.csv
├── notebooks/
│   └── analysis.ipynb
├── powerbi/
│   └── dashboard.pbix
├── screenshots/
│   └── dashboard-preview.png
├── presentation/
│   └── capstone-presentation.pptx
├── README.md
└── requirements.txt
```
## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

</body>
</html>
