import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data from CSV file
movies = pd.read_csv("data/tmdb_5000_movies.csv")

# Keep only movie title and plot summary
movies = movies[['title', 'overview']].dropna()

# Convert plot summaries into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english')
movie_vectors = vectorizer.fit_transform(movies['overview'])

# Create a similarity score matrix
similarity_scores = cosine_similarity(movie_vectors)

# Recommend similar movies
def recommend_movies(movie_name):
    if movie_name not in movies['title'].values:
        print(" Movie not found in the list.")
        return

    index = movies[movies['title'] == movie_name].index[0]
    scores = list(enumerate(similarity_scores[index]))
    top_matches = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    print(f"\nSince, you liked '{movie_name}', you might also like:")
    for i, score in top_matches:
        print(f" {movies.iloc[i]['title']} (Match score: {score:.2f})")

# Ask the user for a movie
user_movie = input("Enter the name of a movie you like: ")
recommend_movies(user_movie)
