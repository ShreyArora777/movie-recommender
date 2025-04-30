
# Recommend a Movie

 You can use Python and the TMDb 5000 Movie Dataset to make this simple "content-based movie recommender system."  It tells the user about movies that are like the one they typed in by looking at themes, keywords, cast.


 The information we used comes from the [TMDb 5000 Movie information on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and is in the file `data/tmdb_5000_movies.csv`.

 ## 🔧  How It Works
 1. Brings the TMDb file into memory.
 2. Cleans and processes columns that matter, like `cast}, `genres`, `keywords}, and so on.
 3: Puts them all together in one "tags" column.
 4. Uses **CountVectorizer** to turn tags into number vectors.
 5. Uses "cosine similarity" to find movies that are most like the one you give it.
 6. Lists the five movies that are most like it.

 ## What You Need
 Do not forget to add the following Python packages:

 Do this: ```bash pip install pandas numpy scikit-learn
