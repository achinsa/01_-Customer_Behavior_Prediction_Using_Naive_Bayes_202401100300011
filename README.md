## Customer Behavior Prediction using Naive Bayes ##
## Project Description
This project focuses on building a machine learning model to classify customers as either Bargain Hunters or Premium Buyers based on their purchasing behavior. It leverages a Naive Bayes classifier, a probabilistic algorithm commonly used for classification tasks.
The classification is performed using features that simulate customer purchase patterns such as average purchase value, discount usage, and brand loyalty. The model is trained and evaluated on a synthetic dataset generated for demonstration purposes. Evaluation metrics like accuracy, precision, recall, and F1-score are used to assess model performance.
A confusion matrix heatmap is also plotted to visualize how well the model distinguishes between the two customer types.

## Technologies Used
Python, 
Scikit-learn, 
Pandas, 
Matplotlib, 
Seaborn. 

## How It Works
Generate synthetic classification dataset using make_classification(), 
Split the dataset into training and test sets, 
Train a Gaussian Naive Bayes classifier, 
Predict customer types on test data, 
Evaluate the model using: 
Confusion Matrix, 
Accuracy, 
Precision, 
Recall, 
F1-Scor, 
Visualize results with a heatmap of the confusion matrix. 

## Final Results
  Confusion Matrix 
  
  ![Screenshot 2025-04-22 114502](https://github.com/user-attachments/assets/0afd6c17-8dad-4296-bf0d-ee59159dbe26)


Predicted Bargain	Predicted Premium 
Actual Bargain	26	4 
Actual Premium	3	17 
 Classification Report 
markdown 
Copy 
Edit 
                  precision    recall  f1-score   support

Bargain Hunter       0.90      0.87      0.88        30
Premium Buyer        0.81      0.85      0.83        20

Overall Accuracy: 86% 

## Logic
We want to predict customer behavior — whether someone is a Bargain Hunter or a Premium Buyer — based on features like purchase amount, brand loyalty, etc.
To solve this, we used a Naive Bayes Classifier, which applies Bayes' Theorem with an assumption of independence among features.

Prediction Logic Summary:
Calculate the probability of the data for each class.
Choose the class with the highest probability.
Output the predicted class label.

## Conclusion 
The Naive Bayes model performs well with an accuracy of 86%, showing strong ability in classifying customers into their respective categories. This approach is lightweight, fast, and effective for customer segmentation tasks in e-commerce or retail analytics.
