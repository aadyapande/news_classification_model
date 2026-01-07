# News Classification Model using  NLP

## Project Overview
This project implements a **complete Natural Language Processing (NLP) pipeline** to classify news articles into **five categories**: Business, Entertainment, Politics, Sport, and Technology.
The model is trained on the **BBC News Archive Dataset** using classical machine learning techniques.
The objective of this project is to demonstrate end-to-end ML pipeline development, including data preprocessing, feature engineering, model training, evaluation, and modular code organization.

## Dataset
Source: BBC News Archive (Kaggle) 

https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive 

Total Samples: 2,225 

Classes: Business, Entertainment, Politics, Sport, Tech 

Each Article contains: Category(label), Filename, Title,Content

## Tech Stack
1. Python
2. Pandas
3. Scikit-learn
4. TF-IDF Vectorizer
5. Logistic Regression
6. Pickle (model persistence)

## Model Used
Logistic Regression 

## Project Pipeline
#### 1. Data Preprocessing
Removed null values

Combined title and content

Text cleaning (lowercasing, removing special characters)

Saved cleaned dataset for reuse

#### 2.Feature Engineering

Converted text data into numerical features using TF-IDF

Removed English stopwords

Limited vocabulary size to 5,000 features

#### 3. Model Training

Algorithm: Logistic Regression

Train–test split (80–20)

Model trained on TF-IDF features

#### 4. Evaluation

Accuracy Score

Confusion Matrix

Metrics saved to file

## Folder Structure
news_classification_model_LR/

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│

├── results/

│

├── src/

│   ├── data_preprocessing.py

│   ├── feature_engineering.py

│   ├── train.py

│   ├── evaluate.py

│   └── config.py

│

├── main.py

└── README.md


## How to Run
1. Open terminal in the **news_classification_model_LR folder**
2. python main.py
This will:
Preprocess data,
Train the model and
Evaluate performance

## Results

Model Accuracy: ~97.3%

The model performs strongly across all news categories.


