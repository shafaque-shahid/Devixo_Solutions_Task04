
# Movie Preference Prediction and Recommendation System

**Devixo Solutions – Machine Learning Internship**  
**Task 04: End-to-End ML Application with Deployment**

## 1. Project Overview

This project is an end-to-end Machine Learning application that predicts movie preferences using historical user ratings and recommends movies that users have not watched or rated before.

The application is developed using Python, Scikit-learn, Pandas, Joblib, and Streamlit.

## 2. Dataset

Dataset: MovieLens Latest-Small

- Total Movies: 9,742
- Total Ratings: 100,836
- Total Users: 610

Files:
- movies.csv
- ratings.csv

Ratings are converted into binary labels:
- Rating >= 4.0: Liked (1)
- Rating < 4.0: Not Liked (0)

## 3. Machine Learning Models

Four classification algorithms were trained and compared:

1. Logistic Regression
2. K-Nearest Neighbors
3. Random Forest
4. Gradient Boosting

Gradient Boosting was selected and optimized using RandomizedSearchCV.

## 4. Final Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 71.57% |
| Precision | 69.52% |
| Recall | 72.98% |
| F1 Score | 71.21% |
| ROC-AUC | 79.01% |

## 5. Application Features

### Home
Displays project information and dataset statistics.

### Movie Recommendations
- Select an existing user.
- Choose the number of recommendations.
- Filter recommendations by genre.
- View predicted preference scores.

### Movie Search
Search movies by title and view release year and genres.

### Dataset Overview
Explore movie and rating records and rating distribution.

### Model Performance
View evaluation metrics and feature importance.

## 6. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## 7. Project Files

```text
Movie-Recommendation-System/
├── app.py
├── movie_recommendation_model (1).pkl
├── movies.csv
├── ratings.csv
├── requirements.txt
└── README.md
