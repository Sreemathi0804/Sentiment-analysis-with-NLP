# Sentiment-analysis-with-NLP

COMPANY: CODTECH IT SOLUTIONS

NAME: SREEMATHI.R

INTERN ID: CT06DH682

DOMAIN: MACHINE LEARNING

DURATION: 6 WEEKS

MENTOR: NEELA SANTOSH

🐦 Sentiment Analysis on Tweets using TF-IDF and Logistic Regression
This project performs sentiment analysis on real-world Twitter data using Natural Language Processing (NLP) techniques. The goal is to classify tweets as positive, negative, or neutral using TF-IDF vectorization and a Logistic Regression model.

📊 Project Overview
Twitter is a valuable source of public sentiment across a wide range of topics — from customer feedback to breaking news. This project leverages a dataset of tweets to build a sentiment classifier using machine learning techniques. The workflow includes:

Preprocessing raw tweet text
Extracting features using TF-IDF
Training a Logistic Regression model
Evaluating and visualizing the model performance

📁 Dataset: Tweets.csv
The dataset used in this project (Tweets.csv) contains thousands of real tweets labeled with sentiment. Each row includes:
text: The tweet content
airline: (if from airline dataset) or topic-related label
sentiment: Sentiment label — typically positive, negative, or neutral
Dataset source: Twitter US Airline Sentiment (or your custom tweet dataset)

🔧 Workflow
Data Cleaning & Preprocessing
Lowercasing
Removing URLs, mentions, hashtags, numbers, and special characters
Tokenization
Removing stopwords (using NLTK)
(Optional) Lemmatization

Feature Extraction
TF-IDF Vectorizer to convert cleaned tweets into numerical features.
Support for unigrams and bigrams.

Model Training
Logistic Regression Classifier
Train/test split using train_test_split
Evaluation Metrics

Accuracy
Precision, Recall, F1-score
Confusion Matrix
ROC-AUC (for binary classification)

Visualizations
Word frequency bar plots
Sentiment distribution pie chart
Confusion matrix heatmap

📚 Libraries Used
pandas, numpy: Data manipulation
nltk, re: Text processing
scikit-learn: Model building and evaluationmatplotlib, seaborn: Visualization

📈 Results & Insights
The model successfully predicts sentiment on tweets with reasonable accuracy.TF-IDF captured significant patterns in tweet language to distinguish sentiment classes.Common negative words include "delay", "cancel", "rude", while positive tweets include "thanks", "great", "love".
