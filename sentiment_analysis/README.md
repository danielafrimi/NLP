## Twitter Toxic Comments Sentiment Analysis

This project uses machine learning to analyze a dataset of toxic comments from Twitter and predict different classes of toxicity.
The goal of the project is to build a model that can accurately predict whether a given comment is toxic or non-toxic, and if it is toxic, what type of toxicity it exhibits.

Getting Started
To get started with this project, you will need to have Python 3.x installed on your computer. 

Dataset
The dataset used in this project is the Twitter toxic comments dataset, which contains a collection of comments from Twitter that have been labeled as toxic or non-toxic, as well as the type of toxicity exhibited by each toxic comment. The dataset is provided Kaggle.

Preprocessing and Feature Engineering
The comments in the dataset are preprocessed and feature engineered before being used to train the machine learning model. 
The preprocessing steps include removing special characters and stopwords, converting the text to lowercase, and stemming the words. 
Additionally, new features are engineered from the text, such as the number of exclamation points, question marks, and mentions in the comment.

Word Clouds
Word clouds are generated for each category of toxicity to visualize the most common words used in each category. 
The word clouds help to identify the types of language used in toxic comments and can aid in the development of new features for the machine learning model.

Spam Detection
Spam detection is performed on the Twitter toxic comments dataset to identify comments that are likely to be spam. 
Spam comments are removed from the dataset to improve the accuracy of the machine learning model.


Machine Learning Model
The machine learning model used in this project is a Logistic Regression, which is trained on the preprocessed comments and their corresponding labels.

Evaluation
The performance of the machine learning model is evaluated using several metrics, including precision, recall, and F1 score. The model achieves a high F1 score on the test set, indicating that it is able to accurately predict the class of toxicity for new comments.

Conclusion
This project demonstrates the use of machine learning for sentiment analysis of toxic comments from Twitter. 
By preprocessing the comments and training a Support Vector Machine classifier, 
we are able to accurately predict the class of toxicity for new comments. 
The model can be further improved by using more advanced preprocessing techniques and exploring other machine learning algorithms.





