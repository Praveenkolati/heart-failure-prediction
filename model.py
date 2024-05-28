import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, accuracy_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Checking for duplicates
def checking_removing_duplicates(df):
    count_dups = df.duplicated().sum()
    print("Number of Duplicates: ", count_dups)
    if count_dups >= 1:
        df.drop_duplicates(inplace=True)
        print('Duplicate values removed!')
    else:
        print('No Duplicate values')

checking_removing_duplicates(df)

# Handling outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Handling class imbalance
live = df_out[df_out.DEATH_EVENT == 0]
die = df_out[df_out.DEATH_EVENT == 1]

die_upsampled = resample(die, replace=True, n_samples=len(live), random_state=0)

upsampled = pd.concat([live, die_upsampled])

# Split data
X = upsampled.drop(['DEATH_EVENT'], axis=1)
y = upsampled['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models
models = [
    LogisticRegression(max_iter=10000, solver='lbfgs'),
    LinearDiscriminantAnalysis(),
    GaussianNB(),
    SVC(probability=True),
    AdaBoostClassifier(algorithm='SAMME'),
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    ExtraTreesClassifier(n_jobs=-1)
]

best_model = None
best_accuracy = 0.0

for model in models:
    if isinstance(model, LogisticRegression):
        # Use a different solver and increase the max_iter
        model = LogisticRegression(max_iter=10000, solver='lbfgs')

    # Check if the model is ExtraTreesClassifier
    if isinstance(model, ExtraTreesClassifier):
        et_model = ExtraTreesClassifier(n_jobs=-1)
        et_model.fit(X, y)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model.__class__.__name__}: {accuracy}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Calibrate the best model
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_train_scaled, y_train)
y_pred_calibrated = calibrated_model.predict(X_test_scaled)

# Performance metrics
print("Classification Report:")
print(classification_report(y_test, y_pred_calibrated))

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_calibrated), annot=True, cmap='YlGnBu', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("fig1")

# ROC AUC curve
probs = calibrated_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("fig2")
# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, calibrated_model.predict_proba(X_test_scaled)[:, 1])

plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig("fig3")

# Save the best model
with open('model.pkl', 'wb') as file:
    pickle.dump(calibrated_model, file)

# Load the model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Predict probabilities for test set
probabilities = loaded_model.predict_proba(X_test_scaled)

# Check predictions
# For example, let's check the first 10 predictions
for i in range(10):
    print(f"Actual: {y_test.iloc[i]}, Predicted probability of dying: {probabilities[i][1]}, Predicted probability of living: {probabilities[i][0]}")

# Cross-validation
scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)

# Print scores and calculate cross-validation score mean
print("Cross-Validation Scores:", scores) 
print("Mean Cross-Validation Score: {},".format(scores.mean()))

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train_scaled, y_train, cv=5)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Learning Curves')
plt.savefig("fig4")

# Grid Search
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}

grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Make predictions on the test set
predictions = grid_search.predict(X_test_scaled)

# Performance metrics
print("Classification Report after Grid Search:")
print(classification_report(y_test, predictions))

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cmap='YlGnBu', fmt='g')
plt.title('Confusion Matrix after Grid Search')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("fig5")

# ROC AUC curve
probs = grid_search.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve after Grid Search')
plt.legend(loc="lower right")
plt.savefig("fig6")
