#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings

# importing relavant packages and classifiers from sklearn library
from sklearn.preprocessing import StandardScaler # for scaling
from sklearn.model_selection import train_test_split # for data splitting

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#load database
df = pd.read_csv("train.csv")
df.head()

#read columns
df.columns

# size of dataset
len(df)

# size of dataset
len(df)

# Calculate class counts and percentages
sns.set_style("darkgrid")
class_counts = df['ACTION'].value_counts()
total_samples = len(df)
percentage_labels = [(count / total_samples) * 100 for count in class_counts]

# Create a bar chart to visualize the class distribution
plt.figure(figsize=(8, 6))
ax = class_counts.plot(kind='bar')
plt.title('Access Control Distribution', fontweight='bold', fontsize=14)
plt.xlabel('Employee Actions', fontweight='bold', fontsize=14)
plt.ylabel('Number of Employee Records', fontweight='bold', fontsize=14)
plt.xticks(rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Add percentage labels on top of the bars (make them bold)
for i, count in enumerate(class_counts):
    ax.text(i, count, f'{percentage_labels[i]:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.show()

# Separating the target label from the features
y = df["ACTION"] # labels
X = df.drop("ACTION", axis=1)  # training features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Scaling Dataset
scaler = StandardScaler()

# Fit and transform the scaler on the training set
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing set using the same scaler
X_test_scaled = scaler.transform(X_test)

#convert y_train to numpy array
y_train = np.array(y_train)

# building a general function for training the model
def model_train(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred_tr = model.predict(X_train)
    y_pred = model.predict(X_test)


    print("--------------------Testing Performance----------------------")
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("=======================================")
    print(" \n ")
    print("============Accuracy==========")
    print(accuracy_score(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("============= Recall ===================")

    # Calculate recall
    TP = conf_matrix[1, 1]  # True Positives
    FN = conf_matrix[1, 0]  # False Negatives
    recall = TP / (TP + FN)

    print(f'Recall: {recall:.5f}')

    print("============= Precision ===================")

    # Calculate precision
    TP = conf_matrix[1, 1]  # True Positives
    FP = conf_matrix[0, 1]  # False Positives
    precision = TP / (TP + FP)

    print(f'Precision: {precision:.5f}')

    print("============= F1 Score ===================")

    # Calculate precision and recall
    TP = conf_matrix[1, 1]  # True Positives
    FP = conf_matrix[0, 1]  # False Positives
    FN = conf_matrix[1, 0]  # False Negatives

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f'F1 Score: {f1_score:.5f}')

    print("============= True Positive Rate (TPR) ===================")

    # Calculate True Positive Rate (TPR)
    TPR = recall

    print(f'True Positive Rate (TPR): {TPR:.5f}')

    print("============= False Positive Rate (FPR) ===================")

    # Calculate False Positive Rate (FPR)
    FP = conf_matrix[0, 1]  # False Positives
    TN = conf_matrix[0, 0]  # True Negatives
    FPR = FP / (FP + TN)

    print(f'False Positive Rate (FPR): {FPR:.5f}')

    return y_pred

model = RandomForestClassifier(n_estimators=100, random_state=42)
y_pred2 = model_train(model, X_train, X_test, y_train, y_test)

from lightgbm import LGBMClassifier
hyperparameters = {
    'learning_rate': 0.5,
    'max_depth': 10,
    'n_estimators': 300
}

# Create the LGBMClassifier with specified hyperparameters
model = LGBMClassifier(**hyperparameters)

#model = LGBMClassifier()
y_pred4 = model_train(model, X_train, X_test, y_train, y_test)