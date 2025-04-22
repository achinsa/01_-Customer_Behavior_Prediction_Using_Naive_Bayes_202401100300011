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
  Confusion Matrix (sample output) 

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
## Conclusion 
The Naive Bayes model performs well with an accuracy of 86%, showing strong ability in classifying customers into their respective categories. This approach is lightweight, fast, and effective for customer segmentation tasks in e-commerce or retail analytics.
