# Customer Churn Classification and Sampling Techniques Report

## 1. Introduction

This project investigates customer churn prediction using machine learning classification techniques and different approaches for addressing class imbalance.

The analysis uses the Telco Customer Churn dataset and focuses primarily on a **Decision Tree Classifier**. Several sampling techniques are applied to the training data to investigate how balancing the classes affects model performance.

The project also includes hyperparameter tuning using **GridSearchCV** to identify a better-performing Decision Tree configuration.

The main objectives are to:

- Inspect and prepare the customer churn dataset.
- Clean and preprocess the data.
- Convert categorical variables into numerical features.
- Split the data into training and testing sets.
- Train a Decision Tree classification model.
- Investigate the effect of different sampling techniques.
- Compare model performance using accuracy, precision, recall and F1-score.
- Tune the Decision Tree hyperparameters.
- Evaluate the final models on unseen test data.

---

# 2. Dataset Description

The notebook loads the dataset using:

```python
df = pd.read_csv('data\\data.csv')
```

The original dataset contains:

- **7,043 records**
- **21 columns**

The variables contain information about customer demographics, services, contracts and billing.

Important variables include:

- `customerID`
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn`

The target variable is **`Churn`**, which identifies whether a customer has left the service.

---

# 3. Data Cleaning

## 3.1 Converting `TotalCharges`

The notebook identifies that `TotalCharges` should be numerical but is initially stored as a non-numeric data type.

The following code was used:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors='coerce'
)
```

Using `errors='coerce'` converts invalid values into missing values (`NaN`).

The dataset subsequently contained **11 missing values in `TotalCharges`**.

## 3.2 Removing Missing Values

The notebook removes rows containing missing values:

```python
df.dropna(how='any', inplace=True)
```

After this operation, the dataset contained:

**7,032 records**

The notebook then confirmed that there were no remaining missing values.

---

# 4. Feature and Target Selection

The customer identifier and target variable were removed from the independent variables:

```python
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']
```

The `customerID` column was excluded because it identifies individual customers rather than representing a useful predictive characteristic.

The resulting feature set contains customer information that can potentially be used to predict churn.

---

# 5. Encoding Categorical Variables

The dataset contains many categorical variables, which machine learning algorithms cannot directly process in their original text format.

The notebook uses Pandas dummy encoding:

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

The `drop_first=True` option removes one category from each categorical variable to avoid redundant dummy variables.

The `dtype='int'` option converts the resulting dummy variables into numerical `0` and `1` values.

This creates a numerical feature matrix suitable for machine learning.

---

# 6. Training and Testing Data

The data was divided into training and testing datasets.

The first train-test split used:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
```

The resulting datasets contained:

- **Training data:** 5,274 records
- **Testing data:** 1,758 records

The notebook then standardised the features using `StandardScaler`:

```python
sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
```

The scaler was fitted on the training data and then applied to the test data.

---

# 7. Initial Decision Tree Model

The first machine learning model used was a Decision Tree Classifier:

```python
from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier()

model_dt.fit(X_train_sc, y_train)

y_pred_dt = model_dt.predict(X_test_sc)
```

The initial model achieved:

**Accuracy: 72.24%**

The classification report was:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| No | 0.82 | 0.80 | 0.81 |
| Yes | 0.47 | 0.49 | 0.48 |

The model therefore performed considerably better at identifying customers who did not churn than customers who did churn.

---

# 8. Class Imbalance

Customer churn is an imbalanced classification problem because there are more customers who did not churn than customers who did.

The later stratified training split showed:

| Class | Training Records |
|---|---:|
| No | 3,872 |
| Yes | 1,402 |

The test set contained:

| Class | Test Records |
|---|---:|
| No | 1,291 |
| Yes | 467 |

This imbalance is important because a model can achieve reasonable overall accuracy while still performing poorly when identifying the minority churn class.

For this reason, the notebook investigates several sampling techniques.

---

# 9. Random Oversampling

The first approach was **RandomOverSampler**.

Random oversampling increases the representation of the minority class by duplicating existing minority-class observations.

The notebook uses:

```python
from imblearn.over_sampling import RandomOverSampler

oversampler = RandomOverSampler(
    sampling_strategy='auto',
    random_state=42,
    shrinkage=None
)
```

The resampled training data was then used to train another Decision Tree.

### Results

**Accuracy: 72.53%**

For customers who churned:

- Precision: 0.47
- Recall: 0.48
- F1-score: 0.48

Compared with the original Decision Tree, random oversampling produced a small improvement in overall accuracy but did not substantially improve churn detection.

---

# 10. Random Undersampling

The notebook next applies **RandomUnderSampler**.

This method reduces the number of majority-class observations to create a more balanced training dataset.

```python
from imblearn.under_sampling import RandomUnderSampler

downsampler = RandomUnderSampler(
    sampling_strategy='auto',
    random_state=42,
    replacement=False
)
```

### Results

**Accuracy: 67.97%**

For the churn class:

- Precision: 0.43
- Recall: 0.68
- F1-score: 0.53

Although overall accuracy decreased, churn recall increased substantially to **0.68**.

This demonstrates an important trade-off: reducing the majority class can result in the model identifying more actual churners, but it can also reduce overall accuracy.

---

# 11. SMOTE

The notebook also applies **SMOTE — Synthetic Minority Over-sampling Technique**.

Unlike simple random oversampling, SMOTE creates synthetic minority-class observations based on neighbouring minority observations.

The notebook uses:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    random_state=42,
    sampling_strategy='auto',
    k_neighbors=5
)
```

### Results

**Accuracy: 71.96%**

For the churn class:

- Precision: 0.47
- Recall: 0.53
- F1-score: 0.50

SMOTE improved churn recall compared with the original model, although overall accuracy was slightly lower.

---

# 12. SMOTEENN

The notebook then applies **SMOTEENN**, a combination of oversampling and data cleaning.

SMOTE creates synthetic minority observations, while Edited Nearest Neighbours removes observations that may be incorrectly positioned or contribute to noisy class boundaries.

### Results

**Accuracy: 70.93%**

For the churn class:

- Precision: 0.46
- Recall: **0.77**
- F1-score: **0.58**

This is a significant result.

The SMOTEENN model achieved the highest churn recall among the first group of sampling experiments.

A recall of **0.77** means the model correctly identified a large proportion of the customers who actually churned.

However, this improvement came with a reduction in overall accuracy.

---

# 13. ADASYN

The notebook also uses **ADASYN (Adaptive Synthetic Sampling)**.

ADASYN generates additional minority-class observations with greater emphasis on areas that are more difficult for the classifier to learn.

### Results

**Accuracy: 71.10%**

For the churn class:

- Precision: 0.45
- Recall: 0.54
- F1-score: 0.49

ADASYN therefore produced a higher churn recall than the original Decision Tree but did not achieve the strong recall observed with SMOTEENN.

---

# 14. AllKNN

The notebook investigates **AllKNN**, an undersampling technique that removes potentially noisy or ambiguous observations from the majority class.

The model was trained after AllKNN resampling.

The reported accuracy was approximately **67.01%**.

However, there is an important issue in this section of the notebook.

The code reports:

```python
print(classification_report(y_test, y_pred_dt4))
```

rather than using:

```python
y_pred_dt7
```

which is the prediction generated by the AllKNN model.

Therefore, the classification report shown in the notebook does **not correspond to the AllKNN prediction**, even though the accuracy calculation uses `y_pred_dt7`.

This should be corrected before treating the AllKNN classification report as a valid result.

---

# 15. Stratified Sampling

The notebook later improves the train-test splitting approach by using stratification:

```python
train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

Stratification helps maintain a similar class distribution between the training and testing datasets.

This is particularly appropriate for the churn problem because the target classes are imbalanced.

---

# 16. AllKNN with a Controlled Decision Tree

A second AllKNN experiment uses a Decision Tree with:

```python
DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)
```

The model achieved:

**Accuracy: 73.55%**

The classification report showed:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| No | 0.91 | 0.71 | 0.80 |
| Yes | 0.50 | 0.81 | **0.62** |

This result is important because the churn recall increased to **0.81**.

The model therefore identified a high proportion of actual churners while maintaining a reasonable overall accuracy.

---

# 17. Tomek Links

The notebook then investigates **Tomek Links**.

Tomek Links identify pairs of observations from different classes that are very close to each other. Removing majority-class observations associated with these links can help clarify the class boundary.

The notebook applies:

```python
tomek_links = TomekLinks(
    sampling_strategy='majority'
)
```

The resulting resampled data is then standardised and used to train a Decision Tree.

---

# 18. Hyperparameter Tuning

The notebook uses **GridSearchCV** to search for better Decision Tree parameters.

The parameter grid was:

```python
param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

Five-fold cross-validation was used:

```python
grid_search = GridSearchCV(
    dtc,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=1
)
```

The best parameters identified were:

```text
max_depth = 5
min_samples_leaf = 4
min_samples_split = 2
```

---

# 19. Tuned Decision Tree Results

The Decision Tree was retrained using the best parameters.

The resulting accuracy was:

**78.33%**

The classification results were:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| No | 0.86 | 0.85 | 0.85 |
| Yes | 0.59 | 0.61 | 0.60 |

The tuned Decision Tree therefore produced a considerable improvement over the original Decision Tree.

The original accuracy was **72.24%**, compared with **78.33%** after tuning.

This represents an improvement of approximately **6.09 percentage points**.

---

# 20. SMOTETomek

The final section of the notebook investigates **SMOTETomek**.

SMOTETomek combines:

1. SMOTE oversampling.
2. Tomek Links cleaning.

The notebook uses:

```python
smote_tomek = SMOTETomek(
    sampling_strategy=0.5,
    random_state=42
)
```

A Decision Tree with `max_depth=5` is then trained on the resampled data.

### Results

The final model achieved:

**Accuracy: 77.70%**

For the churn class:

- Precision: 0.58
- Recall: 0.56
- F1-score: 0.57

Although this model performed reasonably well, it did not outperform the tuned Decision Tree from the Tomek Links experiment.

---

# 21. Overall Comparison

The main reported results can be summarised as follows:

| Technique / Model | Accuracy | Churn Recall | Churn F1 |
|---|---:|---:|---:|
| Original Decision Tree | 72.24% | 0.49 | 0.48 |
| Random Oversampling | 72.53% | 0.48 | 0.48 |
| Random Undersampling | 67.97% | 0.68 | 0.53 |
| SMOTE | 71.96% | 0.53 | 0.50 |
| SMOTEENN | 70.93% | **0.77** | 0.58 |
| ADASYN | 71.10% | 0.54 | 0.49 |
| AllKNN | ~67.01%* | — | — |
| AllKNN + controlled tree | 73.55% | **0.81** | **0.62** |
| Tuned Decision Tree | **78.33%** | 0.61 | 0.60 |
| SMOTETomek | 77.70% | 0.56 | 0.57 |

\*The AllKNN classification report in the earlier experiment contains a code-reference issue and should be recalculated before being used for comparison.

---

# 22. Discussion

The results demonstrate that handling class imbalance can substantially change model behaviour.

The original Decision Tree achieved an accuracy of **72.24%**, but its recall for the churn class was only **0.49**.

This means the model was missing a significant proportion of customers who actually churned.

Sampling techniques changed this balance.

### Best Churn Recall

The controlled Decision Tree using AllKNN achieved the highest reported churn recall of **0.81**.

SMOTEENN also performed strongly, achieving churn recall of **0.77**.

These models may therefore be useful where the primary objective is identifying as many potential churners as possible.

### Best Overall Accuracy

The hyperparameter-tuned Decision Tree achieved the highest accuracy in the experiments reported in the notebook:

**78.33%**

It also achieved a churn F1-score of **0.60**.

This makes it a strong overall candidate when a balance between predictive accuracy and churn-class performance is required.

---

# 23. Business Interpretation

From a customer-retention perspective, identifying customers who are likely to churn can be valuable.

For example, a company could use a churn prediction model to identify customers who may require:

- Customer-service intervention.
- Retention offers.
- Contract reviews.
- Additional technical support.
- Service or pricing reviews.

However, the model should not be selected solely because it has the highest accuracy.

If the business wants to identify as many potential churners as possible, **recall for the churn class** becomes particularly important.

If the business wants a more balanced model with stronger overall predictive performance, the **tuned Decision Tree** provides a stronger result in this notebook.

---

# 24. Strengths of the Project

The notebook demonstrates a broad range of practical machine learning skills.

These include:

- Data loading and inspection.
- Data cleaning.
- Handling missing values.
- Numerical conversion.
- Categorical feature encoding.
- Train-test splitting.
- Stratified sampling.
- Feature standardisation.
- Decision Tree classification.
- Random oversampling.
- Random undersampling.
- SMOTE.
- SMOTEENN.
- ADASYN.
- AllKNN.
- Tomek Links.
- SMOTETomek.
- Hyperparameter optimisation.
- Five-fold cross-validation.
- Classification metrics.

The project is particularly useful because it does not simply train one model. It investigates how different strategies for handling imbalanced data influence classification performance.

---

# 25. Limitations and Areas for Improvement

Several improvements could be made to the notebook.

## 25.1 Use a Consistent Evaluation Strategy

The notebook contains several different experimental setups. A more systematic comparison would use the same:

- Train/test split.
- Stratification method.
- Random state.
- Decision Tree parameters.
- Evaluation metrics.

This would make comparisons between sampling techniques more reliable.

## 25.2 Evaluate More Than Accuracy

Because churn is imbalanced, future evaluation should include:

- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- ROC-AUC.
- Precision-recall AUC.

Particular attention should be given to the **churn class**.

## 25.3 Avoid Potential Data Leakage

Sampling should only be performed on the training dataset, never on the complete dataset before the train-test split.

The later sections of the notebook follow this principle more clearly by splitting the data first and then applying resampling to the training data.

## 25.4 Compare Models Beyond Decision Trees

The notebook focuses primarily on Decision Trees.

Future work could compare the best sampling strategies using:

- Logistic Regression.
- Random Forest.
- Support Vector Machine.
- Gradient Boosting.
- XGBoost.

This would determine whether the benefits of the sampling methods are specific to Decision Trees or generalise across algorithms.

---

# 26. Recommended Next Steps

To develop the project further, the following workflow is recommended:

1. Clean the dataset.
2. Encode the categorical features.
3. Perform a stratified train-test split.
4. Keep the test dataset untouched.
5. Apply each sampling method only to the training data.
6. Train the same Decision Tree configuration for each experiment.
7. Evaluate every model using the same metrics.
8. Use GridSearchCV for hyperparameter optimisation.
9. Compare the final models using a results table.
10. Select the model according to the business objective.
11. Investigate feature importance.
12. Save the final model.
13. Build an interactive churn prediction application.
14. Deploy the application as a portfolio project.

---

# 27. Conclusion

This project provides a practical investigation into **customer churn classification and class-imbalance sampling techniques**.

The initial Decision Tree achieved an accuracy of **72.24%**, with a churn recall of only **0.49**.

Different sampling approaches produced different results. In particular, **SMOTEENN increased churn recall to 0.77**, while the controlled Decision Tree using AllKNN achieved a reported churn recall of **0.81**.

However, the strongest overall accuracy reported in the notebook was achieved by the **hyperparameter-tuned Decision Tree**, which reached **78.33% accuracy** with a churn F1-score of **0.60**.

The project demonstrates that there is no single metric that should always determine model selection. Accuracy, precision, recall and F1-score need to be considered together, particularly when working with an imbalanced classification problem such as customer churn.

Overall, the notebook demonstrates practical knowledge of Python-based data preprocessing, machine learning classification, imbalanced-data techniques and hyperparameter optimisation. It provides a strong foundation for developing a more advanced customer churn prediction application.