# ML Assignment 2 - Online Shoppers Purchase Prediction

M.Tech AIML, BITS Pilani WILP - Machine Learning Assignment 2

GitHub repo: https://github.com/Poornanand95/ML_Assignment_2

Streamlit app: https://poornanand-mlassignment2-e4ws3rl6fuar7v8agtchwh.streamlit.app/

---

## a. Problem statement

When people visit an online store, most of them just browse and leave. Only a small number actually buy
something. In this assignment I try to predict, using the data collected during a browsing session, whether
that session will end in a purchase or not. The target column is `Revenue` (1 = bought something,
0 = did not buy), so it is a binary classification problem.

I trained five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM) on the same
dataset, compared them using six evaluation metrics on the same test set, and made a Streamlit app where
the test data can be uploaded and the saved models can be checked one by one.

## b. Dataset description

**Dataset:** Online Shoppers Purchasing Intention Dataset

**Source:** UCI Machine Learning Repository -
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

- 12,330 rows. Each row is one browsing session, collected over a period of one year.
- 17 input features + 1 target column (`Revenue`)
- No missing values
- The classes are imbalanced: 10,422 sessions (84.5%) did not buy and only 1,908 (15.5%) did buy.
  Because of this I did not rely on accuracy alone and gave more weight to F1, AUC and MCC.

**Features used**

| Feature(s) | Type | What it means |
|---|---|---|
| Administrative, Informational, ProductRelated | numeric | number of pages of that type the user visited |
| Administrative_Duration, Informational_Duration, ProductRelated_Duration | numeric | time spent (in seconds) on those pages |
| BounceRates, ExitRates, PageValues | numeric | Google Analytics values for the pages in the session |
| SpecialDay | numeric | how close the visit was to a special day like Valentine's Day (0 to 1) |
| Month | categorical | month of the visit |
| OperatingSystems, Browser, Region, TrafficType | categorical | stored as number codes, so I treated them as categories |
| VisitorType | categorical | Returning_Visitor / New_Visitor / Other |
| Weekend | binary | whether the visit was on a weekend |

**Preprocessing steps**

1. Split the data 80/20 using a stratified split (random_state = 2611), so both parts have the same
   ratio of buyers. This gives 9,864 rows for training and 2,466 rows for testing. The test part is saved as
   `test_data.csv`.
2. The page counts, durations and PageValues are very skewed (most values are 0 or small and a few are
   very big), so I applied `log1p` to the numeric columns and then `StandardScaler`.
3. One-hot encoding for the categorical columns. Weekend is kept as 0/1.
4. All these steps are inside a scikit-learn `Pipeline` together with the model, so the saved `.joblib`
   file can be used directly on raw test data.
5. For each model I tried a few hyperparameter values with `GridSearchCV` (5-fold stratified CV) and chose
   the best one using F1 score. For Logistic Regression, Decision Tree and SVM I used
   `class_weight="balanced"` because of the class imbalance.

## c. GitHub repository link

https://github.com/Poornanand95/ML_Assignment_2

Files in the repo:

```
app.py               - Streamlit app
train_models.py      - trains and evaluates all 5 models and saves them
shopper_utils.py     - common functions (column names, data cleaning, metrics)
requirements.txt
README.md
test_data.csv        - test split (20%) used for the results below
data/
    online_shoppers_intention.csv   - original dataset from UCI
    train_data.csv                  - training split (80%)
model/
    logistic_regression.joblib
    decision_tree.joblib
    knn.joblib
    naive_bayes.joblib
    svm.joblib
    metrics.json                    - test metrics and best hyperparameters
```

How to run:

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## d. Models used

The five models and the hyperparameters selected by grid search:

| Model | Selected hyperparameters | CV F1 |
|---|---|---|
| Logistic Regression | C = 0.01, class_weight = balanced | 0.6639 |
| Decision Tree | max_depth = 4, min_samples_leaf = 30, class_weight = balanced | 0.6364 |
| KNN | n_neighbors = 31, weights = distance | 0.6205 |
| Naive Bayes (Gaussian) | var_smoothing = 0.1 | 0.5151 |
| SVM (RBF kernel) | C = 0.5, gamma = 0.01, class_weight = balanced | 0.6664 |

### Comparison table (on test data - 2,466 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8710 | 0.9171 | 0.5576 | 0.8115 | 0.6610 | 0.6004 |
| Decision Tree | 0.8617 | 0.9144 | 0.5336 | 0.8534 | 0.6566 | 0.6006 |
| KNN | 0.8974 | 0.9205 | 0.7115 | 0.5681 | 0.6317 | 0.5779 |
| Naive Bayes | 0.7571 | 0.8777 | 0.3766 | 0.8665 | 0.5250 | 0.4559 |
| Support Vector Machine | 0.8763 | 0.9211 | 0.5701 | 0.8194 | 0.6724 | 0.6141 |

Note: SVM was trained without probability output, so its AUC is calculated from the decision function
scores. For the other four models AUC is calculated from predicted probabilities.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed much better than I expected for a simple model. AUC is 0.917 and MCC is 0.600. I think the main reason is that PageValues is a very strong feature (using only PageValues the AUC on the test set is already 0.87), so a linear boundary works quite well. The best C was 0.01, which is strong regularisation, so the extra one-hot columns did not help much. Because of balanced class weights, recall is high (0.81) but precision is only 0.56, meaning it marks some non-buyers as buyers. |
| Decision Tree | The best tree was a small one (depth 4, at least 30 samples in each leaf). Deeper trees gave lower F1 in cross validation, so they were overfitting. Its MCC (0.601) is almost the same as Logistic Regression and recall is good (0.85), but precision is the lowest among the top four models (0.53). The first split of the tree is on PageValues, which again shows it is the most important feature. Easy to explain to someone but a bit less accurate. |
| KNN | Got the highest accuracy (0.897) and the highest precision (0.71), but its recall is low (0.57), so it misses many actual buyers. KNN has no class weight option, and since 85% of the data is non-buyers, most neighbours are non-buyers, so it predicts "buy" only when it is very sure. The high accuracy is mostly because of the imbalance; MCC (0.578) is lower than LR, DT and SVM. k = 31 with distance weighting worked best. |
| Naive Bayes | Weakest of the five models: accuracy 0.757 and MCC 0.456. Gaussian NB assumes the features are independent and normally distributed, which is not true here. For example ProductRelated and ProductRelated_Duration have a correlation of 0.86, and the one-hot columns are just 0/1. It predicts "buy" too often, so recall is the highest (0.87) but precision is only 0.38. It trains very fast though. |
| Support Vector Machine | Best model overall. It has the highest AUC (0.921), F1 (0.672) and MCC (0.614), and a reasonable balance between recall (0.82) and precision (0.57). The RBF kernel can learn a non-linear boundary, which seems to help a little over Logistic Regression. The disadvantage is that it was the slowest to train (around 18 seconds for the grid search, the other models took less than 5 seconds). |
| **Overall, Winner for your dataset?** | **Support Vector Machine.** It is the best on AUC, F1 and MCC, which are the more reliable metrics when the classes are imbalanced. Logistic Regression is very close behind and is faster and easier to interpret, so it would also be a good choice. If we only care about being correct when we predict "buy" (high precision), KNN would be the better option. |

## Streamlit app

Features in the app:

- Upload test data as a CSV file (there is also a button to download `test_data.csv` and a toggle to use the
  copy already in the repo)
- Dropdown to select one of the five models
- Shows Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model
- Confusion matrix, ROC curve and classification report
- "Compare all models" tab which runs all five models on the uploaded file
- "Predictions" tab which shows the prediction for each row, with an option to show only wrong predictions
  and download the results

The app only loads the saved models from the `model/` folder. No training happens inside the app.
