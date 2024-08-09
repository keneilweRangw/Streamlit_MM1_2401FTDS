import streamlit as st
import pandas as pd
import numpy as np
import pickle
import surprise
import base64
from io import BytesIO
from surprise import SVD
import base64
import pickle

import base64
import pickle

def fix_base64_padding(base64_string):
    padding = len(base64_string) % 4
    if padding != 0:
        base64_string += '=' * (4 - padding)
    return base64_string

def load_model_from_base64(base64_string):
    base64_string = fix_base64_padding(base64_string)
    model_data = base64.b64decode(base64_string)
    return pickle.loads(model_data)

# Path to your model file
model_file_path = r'baseline_model.pkl'

# Convert model file to base64 string
with open(model_file_path, 'rb') as file:
    model_base64 = base64.b64encode(file.read()).decode('utf-8')

# Save the base64 string to a file
with open('model_base64.txt', 'w') as text_file:
    text_file.write(model_base64)

# Load base64 string from file
with open('model_base64.txt', 'r') as file:
    model_base64 = file.read()

# Load the model from base64 string
model = load_model_from_base64(model_base64)

# Now you can use the model in your app
print("Model loaded successfully!")

# Define recommendation functions
def get_user_recommendations(favorite_anime_names, model):
    # Retrieve anime data
    anime_df = pd.read_csv(r'cleanedAnime.csv')
    
    # Create a dictionary to map anime names to their IDs
    anime_id_mapping = pd.Series(anime_df['anime_id'].values, index=anime_df['name']).to_dict()

    # Filter out anime IDs that are not in the favorite_anime_names
    favorite_anime_ids = [anime_id_mapping.get(name) for name in favorite_anime_names if anime_id_mapping.get(name) is not None]

    # Generate predictions for each anime
    predictions = []
    for anime_id in anime_df['anime_id']:
        pred = model.predict(uid='user', iid=anime_id)
        predictions.append((anime_id, pred.est))

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get top recommendations
    top_anime_ids = [anime_id for anime_id, _ in predictions[:10]]  # Get top 10 recommendations
    top_anime_names = [anime_df[anime_df['anime_id'] == anime_id]['name'].values[0] for anime_id in top_anime_ids]

    return top_anime_names



def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def get_content_recommendations(favorite_anime_names):
    # Load anime data
    anime_df = pd.read_csv(r'cleanedAnime.csv')
    
    # Create a dictionary to map anime names to their genres (sets)
    anime_genres = anime_df.set_index('name')['genre'].apply(lambda x: set(x.split(','))).to_dict()

    # Calculate similarity scores
    scores = []
    for anime_name, anime_genre in anime_genres.items():
        if anime_name in favorite_anime_names:
            continue
        similarity = np.mean([jaccard_similarity(anime_genre, anime_genres[fav_name]) for fav_name in favorite_anime_names if fav_name in anime_genres])
        scores.append((anime_name, similarity))
    
    # Sort scores and get top recommendations
    scores.sort(key=lambda x: x[1], reverse=True)
    top_anime_names = [anime_name for anime_name, _ in scores[:10]]  # Top 10 recommendations
    
    return top_anime_names

# Add custom CSS to style the app
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000; /* Black background for the main page */
        color: #FFA500; /* Color between yellow and orange */
        font-family: 'Arial', sans-serif; /* Clean and modern font */
    }
    .css-1v3fvcr { 
        background-color: #2C2F33; /* Bright but complementary background for the sidebar */
    }
    .css-1v3fvcr .css-1l7i6wm {
        background-color: #2C2F33; /* Ensure sidebar background color is applied */
    }
    .css-1v3fvcr .css-1d391sg {
        color: #FFA500; /* Color between yellow and orange */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and prepare the dataset
anime_df = pd.read_csv(r'cleanedAnime.csv')
st.title("ANIMEORACLE")
st.write("Your Guide To Discovering Hidden Gems.")
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a page", ["Information Page", "Project Overview", "Recommendation Page"])

if option == "Information Page":
    st.image(r'R.jpeg')  # Using raw string
    st.write("""
        ## Meet the team:
             
           Keneilwe Rangwaga - patricia001105@gmail.com
             
         Muwanwa Tshikovhi - tshikovhimuwanwa@gmail.com
               
          karabo mathibela - karabomathibela44@gmail.com
             
          Koena Mahladisa - kmahladisa9@gmail.com
             
          Mahlatse Lelosa - mahlatselelosa98@gmail.com

             
             """)
    st.write("""Your Streamlit app provides an interactive experience for discovering anime recommendations. Here's an overview of what each section does:

## *Overview of the App*

*ANIMEORACLE* is a personalized anime recommendation system designed to enhance users' anime-watching experience. The app is structured into three main sections:

1. *Information Page*: This section introduces the project team and provides contact details, fostering transparency and connection with users. It also includes a detailed description of the project, emphasizing its significance and objectives in the context of modern recommender systems.

2. *Project Overview*: Here, users can learn about the project's goals and the challenges tackled. The overview explains how the recommender system uses collaborative filtering and content-based filtering methods to suggest anime based on user preferences and title characteristics. This section provides insight into the data analysis, model development, and optimization processes involved.

3. *Recommendation Page*: This interactive feature allows users to input their top three favorite anime and select a recommendation method (collaborative-based or content-based). Users can also upload their own CSV files containing anime data. The app utilizes pre-trained models to offer personalized recommendations based on the provided favorites. Users receive tailored suggestions that align with their tastes, enhancing their discovery of new anime.

*Key Features*:
- *Collaborative-Based Filtering*: Provides recommendations based on user preferences and similarities with other users.
- *Content-Based Filtering*: Suggests anime based on the genres and characteristics of the user's favorite titles.
- *User Interaction*: Allows users to enter their favorite anime and choose a recommendation method.
- *File Upload*: Enables users to upload their own anime data for customized recommendations.

By integrating advanced machine learning techniques, *ANIMEORACLE* aims to offer a highly personalized and enjoyable anime discovery experience. """)



elif option == "Project Overview":
    st.image(r'OIP.jpeg')  # Using raw string

    st.title("Project Overview")
    st.write("""
        In today's digital era, recommender systems play a crucial role in helping people find content that suits their interests. 
        Platforms like Netflix, Amazon Prime, Showmax, and Disney rely on sophisticated algorithms to suggest movies and shows tailored to individual preferences.
        But have you ever wondered how these platforms seem to know your tastes so well?

        This project aims to develop a recommender system specifically for anime titles, leveraging advanced machine learning techniques. The system will help anime fans discover new shows that match their unique preferences, making the viewing experience more enjoyable and personalized.

        #### Problem Statement
             
        The primary goal of this project is to create a recommender system that can accurately predict how users will rate anime titles they haven't watched yet. We will achieve this by combining collaborative filtering and content-based filtering methods, using a comprehensive dataset from myanimelist.net.

        #### Key challenges we will address include:

        * Data Analysis and Preparation: Cleaning and organizing the dataset to ensure it's suitable for building our models.

        * Collaborative Filtering: Developing a model that recommends anime based on the preferences of users with similar tastes.

        * Content-Based Filtering: Creating a model that suggests anime based on the characteristics of the titles a user has already enjoyed.

        * Evaluation and Optimization: Testing our system and fine-tuning it to provide the best possible recommendations.

        By tackling these challenges, we'll create a powerful recommender system that not only helps users discover new anime but also enhances their overall viewing experience. This project highlights the impact of machine learning in revolutionizing how we find and enjoy content in the entertainment industry.
    """)
    
elif option == "Recommendation Page":    
    # Upload an image
    st.image(r'itachi.jpeg')  # Using raw string
    
    # Upload a CSV file
    uploaded_file = st.file_uploader("Choose a CSV file with anime data", type="csv")
    
    if uploaded_file is not None:
        anime_df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV:")
        st.write(anime_df.head())
    
    # Choose filtering method
    st.write("### Choose Filtering Method")
    filter_method = st.radio(
        "Select a filtering method:",
        options=["Collaborative-Based Filtering", "Content-Based Filtering"]
    )
    
    # Enter favorite anime names
    st.write("### Share Your Top 3 Anime Favorites 🌟")
    
    anime_1 = st.text_input("1️⃣ What's your absolute favorite anime?", "")
    anime_2 = st.text_input("2️⃣ Tell us about your second favorite anime!", "")
    anime_3 = st.text_input("3️⃣ Finally, your third favorite anime?", "")
    
    favorite_anime_names = [anime_1, anime_2, anime_3]
    
    if any(name.strip() == "" for name in favorite_anime_names):
        st.warning("Please fill in all three favorite anime fields.")
    else:
        if st.button("Get Recommendations"):
            if filter_method == "Collaborative-Based Filtering":
                recommendations = get_user_recommendations(favorite_anime_names, model)
            elif filter_method == "Content-Based Filtering":
                recommendations = get_content_recommendations(favorite_anime_names)
            
            st.write("### Recommended Anime Based on Your Favorites:")
            st.write(recommendations)