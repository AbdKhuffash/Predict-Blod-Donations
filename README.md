## Overview
The project aimed to predict blood donation behavior using various machine learning algorithms. The dataset contained information on past donation history, which was used to predict whether an individual would donate blood in March 2007. The project's primary goal was to compare different algorithms' performance in classifying individuals into donors and non-donors.

## Dataset
The dataset used in this project can be found on DrivenData. It contains information on past blood donation history.

## Tools and Libraries Used
The following tools and libraries were integral to the project:

Python: The primary programming language.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations on arrays.

Scikit-learn: Provided machine learning algorithms and utilities for model training, evaluation, and hyperparameter tuning.

Matplotlib and Seaborn: For data visualization.

Jupyter Notebook: For interactive development and documentation.

Google Colab: The project was done via Google Colab, a cloud-based service that provides a Python programming environment.

## Methodology
Data Preprocessing: The dataset was loaded and preprocessed using Pandas. Missing values were removed, and the dataset was described to understand its characteristics. The target variable was renamed for clarity.

Feature Scaling: MinMaxScaler was used to normalize the feature set to ensure that each feature contributes equally to the model training process.

Model Training: Three different models were trained and evaluated: Decision Tree Classifier, Naive Bayes, and Support Vector Classifier (SVC). Additionally, an Artificial Neural Network (ANN) model was trained.

Hyperparameter Tuning: Each model's hyperparameters were tuned using GridSearchCV to find the best-performing combination.

Model Evaluation: The models were evaluated based on accuracy, precision, recall, and F1-score. Confusion matrices and ROC curves were plotted to visualize the models' performance.

## Results
#### Decision Tree (DT)
20% Testing (80% Training)
Accuracy: 76.67%
Precision (Class 1): 0.52
Recall (Class 1): 0.31
F1-Score (Class 1): 0.39
25% Testing (75% Training)
Accuracy: 70.59%
Precision (Class 1): 0.38
Recall (Class 1): 0.33
F1-Score (Class 1): 0.35
30% Testing (70% Training)
Accuracy: 74.22%
Precision (Class 1): 0.42
Recall (Class 1): 0.20
F1-Score (Class 1): 0.27
#### Naive Bayes (NB)
20% Testing (80% Training)
Accuracy: 76.67%
Precision (Class 1): 0.52
Recall (Class 1): 0.31
F1-Score (Class 1): 0.39
25% Testing (75% Training)
Accuracy: 70.59%
Precision (Class 1): 0.38
Recall (Class 1): 0.33
F1-Score (Class 1): 0.35
30% Testing (70% Training)
Accuracy: 74.22%
Precision (Class 1): 0.42
Recall (Class 1): 0.20
F1-Score (Class 1): 0.27
#### Support Vector Machine (SVC)
20% Testing (80% Training)
Accuracy: 78.67%
Precision (Class 1): 0.59
Recall (Class 1): 0.36
F1-Score (Class 1): 0.45
25% Testing (75% Training)
Accuracy: 81.28%
Precision (Class 1): 0.73
Recall (Class 1): 0.36
F1-Score (Class 1): 0.48
30% Testing (70% Training)
Accuracy: 76.44%
Precision (Class 1): 0.60
Recall (Class 1): 0.06
F1-Score (Class 1): 0.10
#### Artificial Neural Network (ANN)
20% Testing (80% Training)
Accuracy: 79.33%
Precision (Class 1): 0.62
Recall (Class 1): 0.36
F1-Score (Class 1): 0.46
25% Testing (75% Training)
Accuracy: 81.28%
Precision (Class 1): 0.71
Recall (Class 1): 0.38
F1-Score (Class 1): 0.49
30% Testing (70% Training)
Accuracy: 81.33%
Precision (Class 1): 0.73
Recall (Class 1): 0.35
F1-Score (Class 1): 0.48

Accuracy: The Artificial Neural Network (ANN) consistently shows the highest accuracy across all test sizes, with its performance peaking at 30% testing data. The accuracy of other models tends to fluctuate, with SVC also showing relatively high accuracy, especially when the test size is increased to 25%.

Precision, Recall, and F1-Score: Generally, ANN and SVC show better performance compared to Decision Tree and Naive Bayes. The precision, recall, and f1-score  tend to improve or remain more stable in ANN and SVC as the size of the test data increases.

Impact of Test Size: Increasing the test size from 20% to 30% tends to result in a slight decrease in performance for DT and NB, particularly in precision and recall. However, SVC and ANN show an improvement or stability.

In conclusion, ANN emerges as the most robust model across different test sizes, showing the highest accuracy and balanced performance metrics. SVC also performs well, particularly in handling the minority class as the test size increases. Decision Tree and Naive Bayes show more sensitivity to changes in test size, particularly in their ability to correctly predict outcomes.


## Conclusion
The project showcased the utility of different machine learning algorithms in solving classification problems, particularly in predicting blood donation behavior. It highlighted the importance of data preprocessing, feature scaling, model selection, hyperparameter tuning, and evaluation metrics in building effective predictive models. This project demonstrated the practical application of machine learning techniques in solving real-world problems, specifically in the healthcare domain.

