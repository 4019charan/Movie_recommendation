import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the datasets
@st.cache_data
def load_data():
    credits_file_path = '/Users/charanbasireddy/Documents/Courses/Software-AI/Final Project/Week 2/tmdb_5000_credits.csv'
    movies_file_path = '/Users/charanbasireddy/Documents/Courses/Software-AI/Final Project/Week 2/tmdb_5000_movies.csv'

    credits = pd.read_csv(credits_file_path)
    movies = pd.read_csv(movies_file_path)
    credits.columns = ['id', 'title', 'cast', 'crew']
    movies = movies.merge(credits, on='id')
    movies = movies[['id', 'title_x', 'genres', 'keywords', 'overview', 'cast', 'crew']]
    movies.rename(columns={'title_x': 'title'}, inplace=True)
    return movies

movies = load_data()

# Preprocess and Feature Engineering
@st.cache_data
def preprocess_data(movies):
    import ast

    def parse_features(x):
        try:
            return [d['name'] for d in ast.literal_eval(x)]
        except:
            return []

    def get_director(x):
        try:
            for crew_member in ast.literal_eval(x):
                if crew_member['job'] == 'Director':
                    return crew_member['name']
            return ''
        except:
            return ''

    movies['genres'] = movies['genres'].apply(parse_features)
    movies['keywords'] = movies['keywords'].apply(parse_features)
    movies['cast'] = movies['cast'].apply(parse_features)
    movies['director'] = movies['crew'].apply(get_director)

    # Limit the number of cast members and keywords
    movies['cast'] = movies['cast'].apply(lambda x: x[:3])
    movies['keywords'] = movies['keywords'].apply(lambda x: x[:5])

    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    movies['genres'] = movies['genres'].apply(clean_data)
    movies['keywords'] = movies['keywords'].apply(clean_data)
    movies['cast'] = movies['cast'].apply(clean_data)
    movies['director'] = movies['director'].apply(clean_data)

    def create_soup(x):
        return ' '.join(x['genres']) + ' ' + ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director']

    movies['soup'] = movies.apply(create_soup, axis=1)
    return movies

movies = preprocess_data(movies)

# Vectorization and Similarity
@st.cache_data
def compute_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = compute_similarity(movies)

# Map movie indices
movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()



def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        # Use fuzzy matching to find the closest title
        best_match, score = process.extractOne(title, indices.index, scorer=fuzz.ratio)
        
        if score < 50:  # Arbitrary threshold for matching; adjust as needed
            return [f"Movie '{title}' not found in the dataset."]
        
        print(f"Best match: {best_match} (Score: {score})")  # Debugging line
        
        # Get the index of the matched movie
        idx = indices[best_match]
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx].tolist()))
        
        # Sort by similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Exclude the input movie and get top 10 recommendations
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the top 10 most similar movies
        return movies['title'].iloc[movie_indices].tolist()
    except Exception as e:
        return [f"An error occurred: {str(e)}"]






# Streamlit App Layout
st.title("🎥 Movie Recommendation System")
st.subheader("Find movies you will love based on your favorites!")

# Movie Input
movie_title = st.text_input("Enter a movie title (e.g., 'The Dark Knight'):")

if st.button("Get Recommendations"):
    if movie_title:
        recommendations = get_recommendations(movie_title)
        st.subheader("Recommended Movies:")
        for idx, rec in enumerate(recommendations):
            st.write(f"{idx+1}. {rec}")
    else:
        st.warning("Please enter a movie title.")

# Footer
st.markdown("---")
st.write("Developed by Charan Basireddy using Streamlit.")
