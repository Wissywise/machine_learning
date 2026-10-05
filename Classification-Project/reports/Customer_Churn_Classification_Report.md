# Customer Churn Classification and Machine Learning Report

## 1. Introduction

This project investigates the use of machine learning classification algorithms to predict whether a telecommunications customer is likely to churn. Customer churn refers to a customer discontinuing their service.

The project uses the **Telco Customer Churn** dataset and applies a complete machine-learning workflow, including data loading, data inspection, data cleaning, feature encoding, feature scaling, training and testing, model development and performance evaluation.

Several classification algorithms were tested, including k-Nearest Neighbours (kNN), Decision Tree, Random Forest, Naive Bayes, Support Vector Machine (SVM), Logistic Regression, AdaBoost, Gradient Boosting and XGBoost.

The purpose of comparing several models was to identify which algorithm provided the strongest classification performance on the test data.

---

## 2. Dataset

The dataset initially contained **7,043 customer records and 21 columns**.

The variables included customer demographic information, service information, contract details and billing information.

Examples of the variables included:

- Customer ID
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Technical Support
- Streaming services
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges
- Churn

The target variable was **Churn**, which contained two possible outcomes:

- `Yes` – the customer churned
- `No` – the customer did not churn

---

## 3. Data Cleaning and Preparation

The dataset was inspected using Pandas, including the `head()` and `info()` functions.

During the inspection, the `TotalCharges` column was identified as being stored as a string rather than a numerical value. This was converted into a numerical format using:

```python
df.TotalCharges = pd.to_numeric(
    df.TotalCharges,
    errors='coerce'
)
```

Using `errors='coerce'` converted values that could not be interpreted as numbers into missing values.

The resulting missing values were then removed:

```python
df.dropna(how='any', inplace=True)
```

This reduced the dataset from **7,043 records to 7,032 records**, meaning that 11 records were removed.

The notebook also examined the distribution of the target variable. Approximately:

- **73.42%** of customers did not churn.
- **26.58%** of customers churned.

This indicates that the dataset is imbalanced, with substantially more non-churn customers than churn customers.

---

## 4. Feature and Target Variables

The customer ID was removed because it does not provide useful predictive information.

The independent variables were created using:

```python
X = df.drop(['customerID', 'Churn'], axis=1)
```

The dependent variable was:

```python
y = df['Churn']
```

The target values were converted from categorical values to numerical values:

```python
y = y.map({'Yes': 1, 'No': 0})
```

Therefore:

- `0` represents a customer who did not churn.
- `1` represents a customer who churned.

---

## 5. Categorical Feature Encoding

Many variables in the dataset were categorical, such as gender, contract type, internet service and payment method.

These variables were converted into numerical features using Pandas dummy encoding:

```python
X = pd.get_dummies(
    X,
    columns=[
        'gender',
        'Partner',
        'Dependents',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaperlessBilling',
        'PaymentMethod'
    ],
    drop_first=True,
    dtype='int'
)
```

After encoding, the dataset contained **30 features**.

The use of `drop_first=True` reduced redundant dummy variables and helped avoid unnecessary duplication of categorical information.

---

## 6. Training and Testing Data

The dataset was divided into training and testing sets using `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
```

This produced:

- **5,274 training records**
- **1,758 testing records**

The test data therefore represented 25% of the dataset.

The use of `random_state=42` ensured that the same split could be reproduced when the notebook was executed again.

---

## 7. Feature Scaling

The project used `StandardScaler` to standardise the feature values:

```python
sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
```

The scaler was fitted using the training data and then applied to both the training and testing data.

This was particularly useful for algorithms such as kNN and SVM because these algorithms can be affected by differences in feature scale.

---

# 8. Machine Learning Models

Several classification algorithms were developed and evaluated.

## 8.1 k-Nearest Neighbours

The first model was a k-Nearest Neighbours classifier:

```python
model = KNeighborsClassifier(
    n_neighbors=10,
    metric='minkowski',
    p=2
)
```

The model achieved an accuracy of:

**77.53%**

The classification report showed an F1-score of **0.51 for churned customers**.

The model performed considerably better at identifying customers who did not churn than customers who churned.

---

## 8.2 Decision Tree

A Decision Tree classifier was then trained:

```python
model_dt = DecisionTreeClassifier()
```

The model achieved an accuracy of:

**71.33%**

The churn class had an F1-score of **0.46**.

This was the lowest accuracy among the main models tested in the notebook.

---

## 8.3 Random Forest

The project then introduced Random Forest:

```python
model_rf = RandomForestClassifier(
    n_estimators=200
)
```

Random Forest is an **ensemble learning algorithm** because it combines predictions from multiple decision trees.

The model achieved:

**78.27% accuracy**

The churn class achieved an F1-score of **0.53**.

Random Forest therefore performed better than the individual Decision Tree model.

---

## 8.4 Naive Bayes

A Bernoulli Naive Bayes classifier was also tested:

```python
model_nb = BernoulliNB()
```

It achieved:

**71.84% accuracy**

However, an important result was that its recall for the churn class was **0.79**.

This means that although its overall accuracy was relatively low, it identified a relatively large proportion of the customers who actually churned.

This demonstrates why accuracy should not be considered the only performance measure for a classification problem.

---

## 8.5 Support Vector Machine

The Support Vector Machine model was created using:

```python
model_svm = SVC()
```

The model achieved:

**79.29% accuracy**

The churn class had:

- Precision: **0.64**
- Recall: **0.48**
- F1-score: **0.55**

The SVM therefore performed strongly compared with the other traditional classifiers.

---

## 8.6 Logistic Regression

Logistic Regression was also evaluated:

```python
model_lr = LogisticRegression()
```

The model achieved:

**79.01% accuracy**

For the churn class, it achieved an F1-score of **0.56**.

This was a strong result and demonstrates that a relatively simple classification algorithm can perform competitively on this dataset.

---

# 9. Ensemble Learning

The project also investigated ensemble-learning methods.

Ensemble learning combines multiple models or sequentially developed learners to improve predictive performance.

Three ensemble approaches were included:

- AdaBoost
- Gradient Boosting
- XGBoost

---

## 9.1 AdaBoost

The AdaBoost model was configured as:

```python
model_ab = AdaBoostClassifier(
    n_estimators=200,
    learning_rate=1,
    estimator=None,
    random_state=42
)
```

The model achieved:

**79.24% accuracy**

The F1-score for the churn class was **0.55**.

AdaBoost performed slightly better than the individual kNN, Decision Tree and Naive Bayes models.

---

## 9.2 Gradient Boosting

Gradient Boosting was configured as:

```python
model_gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=1,
    max_depth=1,
    random_state=42
)
```

The model achieved:

**79.35% accuracy**

The churn class had an F1-score of **0.56**.

This was slightly better than AdaBoost and Logistic Regression in terms of overall accuracy.

---

## 9.3 XGBoost

The final model was XGBoost:

```python
model_xgb = XGBClassifier(
    n_estimators=90,
    learning_rate=1,
    max_depth=1,
    random_state=42
)
```

XGBoost achieved the highest accuracy in the notebook:

**80.09%**

Its classification results for the churn class were:

- Precision: **0.64**
- Recall: **0.53**
- F1-score: **0.58**

XGBoost therefore provided the strongest overall performance among the models tested in this project.

---

# 10. Model Comparison

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 |
|---|---:|---:|---:|---:|
| kNN | 77.53% | 0.59 | 0.45 | 0.51 |
| Decision Tree | 71.33% | 0.45 | 0.48 | 0.46 |
| Random Forest | 78.27% | 0.61 | 0.46 | 0.53 |
| Naive Bayes | 71.84% | 0.48 | 0.79 | 0.60 |
| SVM | 79.29% | 0.64 | 0.48 | 0.55 |
| Logistic Regression | 79.01% | 0.62 | 0.52 | 0.56 |
| AdaBoost | 79.24% | 0.63 | 0.48 | 0.55 |
| Gradient Boosting | 79.35% | 0.63 | 0.51 | 0.56 |
| **XGBoost** | **80.09%** | **0.64** | **0.53** | **0.58** |

Based on the notebook's test results, **XGBoost achieved the highest overall accuracy at 80.09%**.

However, Naive Bayes achieved the highest recall for the churn class at **0.79**. This distinction is important because a business may prefer a model that identifies more potential churners rather than simply maximising overall accuracy.

---

# 11. Individual Customer Prediction

The notebook also demonstrates how the trained kNN model can be used to make a prediction for an individual customer.

A new customer record was created using a DataFrame containing the same 30 features used during model training.

The new data was scaled using the existing StandardScaler:

```python
data_sc = sc.transform(data)
```

The model then generated a prediction:

```python
single_pred = model.predict(data_sc)
```

The output was:

```text
[0]
```

This indicates that the kNN model predicted that the customer would **not churn**.

The notebook also generated prediction probabilities:

```text
[[0.9, 0.1]]
```

This represents the model's estimated probabilities for the two classes, with a higher probability assigned to class `0`.

---

# 12. Evaluation of the Results

The results demonstrate that different classification algorithms behave differently on the same dataset.

The highest overall accuracy was achieved by XGBoost at **80.09%**. However, the results also demonstrate why accuracy alone should not determine which model should be selected.

The dataset contains a class imbalance, with approximately 73.42% of customers not churning and 26.58% churning. Consequently, a model could obtain a reasonably high accuracy by being better at predicting the majority class.

For a real-world customer-retention application, **recall, precision and F1-score for the churn class are particularly important**. Missing a customer who is likely to churn could result in a lost business opportunity.

The Naive Bayes model is an interesting example because its overall accuracy was only 71.84%, but its churn recall was 0.79. This means it identified more of the actual churn cases than the other models tested.

XGBoost provided a better balance of overall accuracy and churn-class performance, achieving an accuracy of 80.09%, precision of 0.64, recall of 0.53 and F1-score of 0.58 for churn.

---

# 13. Strengths of the Project

The project demonstrates several important machine-learning skills:

1. **Data inspection** using Pandas.
2. **Data cleaning** and handling of missing values.
3. **Data type conversion** using `pd.to_numeric()`.
4. **Categorical feature encoding** using dummy variables.
5. **Feature and target separation**.
6. **Train/test splitting**.
7. **Feature scaling** using StandardScaler.
8. Development of multiple classification algorithms.
9. Use of **ensemble learning** techniques.
10. Evaluation using accuracy and classification reports.
11. Prediction of individual customer outcomes.
12. Comparison of several machine-learning approaches.

The project therefore demonstrates a complete end-to-end introductory machine-learning classification workflow.

---

# 14. Limitations and Possible Improvements

Although the project provides useful results, several improvements could make the analysis more robust.

### Class imbalance

The target variable is imbalanced, with significantly more non-churn than churn customers.

Future versions could investigate:

- Class weights
- SMOTE
- Random under-sampling
- Threshold optimisation
- Precision-recall analysis

### Hyperparameter tuning

Most models use manually selected parameters. The project could be improved by using:

```python
GridSearchCV
```

or:

```python
RandomizedSearchCV
```

to identify better-performing hyperparameters.

### Cross-validation

The models are evaluated using one train/test split. Cross-validation could provide a more reliable estimate of model performance.

### Additional evaluation metrics

Future analysis could include:

- Confusion matrix
- ROC-AUC
- Precision-recall curve
- ROC curve
- Matthews correlation coefficient

### Feature importance

For tree-based models such as Random Forest and XGBoost, feature importance could be analysed to identify which customer characteristics contribute most to churn predictions.

### Reproducibility

Random states could be specified consistently across all models, particularly Random Forest, to make results easier to reproduce.

---

# 15. Conclusion

This project successfully developed and compared several machine-learning classification models for predicting customer churn.

The workflow began with data inspection and cleaning, followed by conversion of the `TotalCharges` variable, removal of missing records, separation of independent and dependent variables, categorical encoding and feature scaling.

Nine classification approaches were evaluated. The results showed that **XGBoost achieved the highest overall accuracy of 80.09%**, while Naive Bayes achieved the highest churn recall of 0.79.

The project also demonstrated the value of ensemble learning through Random Forest, AdaBoost, Gradient Boosting and XGBoost.

Overall, the project provides practical evidence of the ability to prepare real-world data, build machine-learning classification models, evaluate their performance and compare different algorithms. Further improvements involving cross-validation, hyperparameter optimisation, class-imbalance techniques and additional evaluation metrics would make the model more robust and suitable for a more advanced production-oriented analysis.