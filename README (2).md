# 📊 Digital Content Attention Analyzer

A Machine Learning based application that analyzes digital learning behavior and predicts content completion probability.

## 🎯 Project Objective

The system analyzes learner behavior such as:

- Watch Time
- Pause Count
- Replay Count
- Notes Count
- Quiz Score
- Login Frequency
- Session Duration
- Previous Content Completion

It predicts:

1. Completion Status
2. Completion Probability
3. Engagement Level
4. User Behavior Segment

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SQL
- SQLite
- Scikit-learn
- Random Forest
- K-Means
- Streamlit

## 🤖 Machine Learning

Supervised Learning algorithms:

- KNN
- Naive Bayes
- Decision Tree
- Random Forest

Evaluation metrics:

- Accuracy
- Precision
- Recall

## 👥 User Segmentation

K-Means clustering is used to identify:

- Passive Users
- Regular Learners
- Highly Engaged Learners

The clustering uses:

- Watch Percentage
- Replay Count
- Quiz Score
- Login Frequency
- Session Duration

## 📊 Features

### Data Processing
- Missing value handling
- Duplicate removal
- Invalid value correction

### Feature Engineering
- Watch Percentage
- Average Session Time
- Engagement Score
- Replay Ratio
- Quiz Performance

### SQL
SQLite database with:

- users
- content
- watch_history
- quiz_results
- login_activity

SQL concepts demonstrated:

- SELECT
- WHERE
- GROUP BY
- HAVING
- CASE
- Aggregate Functions
- JOIN
- Subqueries

## 🚀 Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```