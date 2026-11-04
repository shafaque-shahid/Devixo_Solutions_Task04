import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import re
from sklearn.base import BaseEstimator, TransformerMixin


# Custom transformer must be defined before loading model
class LeakageSafeMovieFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, genre_columns=None, n_splits=5, random_state=42):
        self.genre_columns = genre_columns
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        user_stats = self.user_stats_.reindex(X["userId"].values)
        movie_stats = self.movie_stats_.reindex(X["movieId"].values)

        result = pd.DataFrame(index=X.index)

        result["user_avg_rating"] = user_stats["user_avg_rating"].fillna(self.global_rating_).values
        result["user_rating_count"] = user_stats["user_rating_count"].fillna(0).values
        result["user_like_rate"] = user_stats["user_like_rate"].fillna(self.global_like_).values
        result["movie_avg_rating"] = movie_stats["movie_avg_rating"].fillna(self.global_rating_).values
        result["movie_rating_count"] = movie_stats["movie_rating_count"].fillna(0).values
        result["movie_like_rate"] = movie_stats["movie_like_rate"].fillna(self.global_like_).values

        for genre in self.genre_columns:
            result[genre] = X[genre].fillna(0).values if genre in X.columns else 0

        return result


# Load model and data
@st.cache_resource
def load_model():
    return joblib.load("movie_recommendation_model (1).pkl")

@st.cache_data
def load_data():
    return pd.read_csv("movies.csv"), pd.read_csv("ratings.csv")

model = load_model()
movies, ratings = load_data()

feature_transformer = model.named_steps["features"]
genre_columns = feature_transformer.genre_columns


# Prepare movie features
genre_data = movies["genres"].fillna("").str.get_dummies(sep="|")
genre_data = genre_data.drop(columns=["(no genres listed)"], errors="ignore")
genre_data = genre_data.reindex(columns=genre_columns, fill_value=0)

movie_features = pd.concat(
    [movies[["movieId"]].reset_index(drop=True),
     genre_data.reset_index(drop=True)], axis=1
)


# Extract release year
def get_year(title):
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return match.group(1) if match else "Unknown"

movies["Year"] = movies["title"].apply(get_year)


# Recommendation function
def recommend_movies(user_id, top_n=10, selected_genre="All"):
    watched = ratings.loc[
        ratings["userId"] == user_id, "movieId"
    ].unique()

    candidates = movie_features[
        ~movie_features["movieId"].isin(watched)
    ].copy()

    if selected_genre != "All":
        candidates = candidates[
            candidates[selected_genre] == 1
        ]

    if candidates.empty:
        return pd.DataFrame()

    candidates.insert(0, "userId", user_id)
    candidates["Like Score"] = model.predict_proba(candidates)[:, 1]

    results = candidates.nlargest(top_n, "Like Score")

    results = results.merge(
        movies[["movieId", "title", "genres", "Year"]],
        on="movieId", how="left"
    )

    results["Predicted Preference"] = (
        results["Like Score"] * 100
    ).round(2).astype(str) + "%"

    return results[
        ["title", "Year", "genres", "Predicted Preference"]
    ].rename(columns={
        "title": "Movie",
        "Year": "Release Year",
        "genres": "Genres"
    })


# Sidebar navigation
st.sidebar.title("🎬 Movie AI")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Movie Recommendations", "Movie Search",
     "Dataset Overview", "Model Performance"]
)


# HOME
if page == "Home":
    st.title("🎬 Movie Recommendation System")
    st.write(
        "An end-to-end Machine Learning application that predicts "
        "movie preferences and recommends movies using historical ratings."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Movies", len(movies))
    c2.metric("Total Ratings", len(ratings))
    c3.metric("Total Users", ratings["userId"].nunique())

    st.subheader("About the Project")
    st.write(
        "The application uses a tuned Gradient Boosting classifier "
        "to predict movie preferences and generate personalized recommendations."
    )

    st.info("Explore movie recommendations, search movies, and view model performance using the sidebar.")


# RECOMMENDATIONS
elif page == "Movie Recommendations":
    st.title("🍿 Personalized Movie Recommendations")

    user_ids = sorted(ratings["userId"].unique())
    user_id = st.selectbox("Select User ID", user_ids)

    col1, col2 = st.columns(2)

    with col1:
        top_n = st.slider("Number of recommendations", 5, 20, 10)

    with col2:
        selected_genre = st.selectbox(
            "Filter by genre", ["All"] + genre_columns
        )

    if st.button("Generate Recommendations"):
        results = recommend_movies(user_id, top_n, selected_genre)

        if results.empty:
            st.warning("No movies found for this selection.")
        else:
            st.subheader(f"Recommendations for User {user_id}")
            st.dataframe(results, use_container_width=True, hide_index=True)
            st.caption("Preference scores are model predictions, not guaranteed outcomes.")


# MOVIE SEARCH
elif page == "Movie Search":
    st.title("🔎 Search Movies")

    search = st.text_input("Enter a movie title")

    if search:
        matches = movies[
            movies["title"].str.contains(search, case=False, na=False)
        ]

        st.write(f"Found {len(matches)} movies.")

        if not matches.empty:
            st.dataframe(
                matches[["title", "Year", "genres"]].head(50),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No matching movies found.")


# DATASET
elif page == "Dataset Overview":
    st.title("📊 Dataset Overview")

    st.subheader("Movies")
    st.dataframe(movies.head(20), use_container_width=True)

    st.subheader("Ratings")
    st.dataframe(ratings.head(20), use_container_width=True)

    st.subheader("Rating Distribution")
    fig, ax = plt.subplots()
    ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Ratings")
    st.pyplot(fig)
    plt.close(fig)


# MODEL PERFORMANCE
elif page == "Model Performance":
    st.title("📈 Model Performance")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        "Score": [0.7157, 0.6952, 0.7298, 0.7121, 0.7901]
    })

    st.dataframe(metrics, use_container_width=True)
    st.bar_chart(metrics.set_index("Metric"))

    st.subheader("Top 10 Feature Importance")

    importance = model.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_transformer.transform(
            movie_features.head(1).assign(userId=1)
        ).columns,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    st.bar_chart(importance_df.head(10).set_index("Feature"))


st.sidebar.markdown("---")
st.sidebar.caption("Devixo Solutions ML Internship")
