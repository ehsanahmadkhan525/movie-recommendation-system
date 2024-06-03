import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(
    page_title="Anime Recommender", layout="wide", initial_sidebar_state="expanded"
)
# Load the data
anime_df = pd.read_csv("data/df_anime_sample.csv")
users_ratings_df = pd.read_csv("data/df_users_with_ratings_sample.csv")

# Combine genres and synopsis into a single string for each anime
anime_df["content"] = anime_df["Genres"] + " " + anime_df["sypnopsis"]

# Create a TF-IDF Vectorizer to convert the content into vectors
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(anime_df["content"])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a mapping of anime titles to indices
anime_indices = pd.Series(anime_df.index, index=anime_df["Name"]).drop_duplicates()


# Function to get recommendations based on content
def get_recommendations(anime_name, cosine_sim=cosine_sim):
    # Get the index of the anime that matches the name
    idx = anime_indices[anime_name]

    # Get the pairwise similarity scores of all animes with that anime
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the animes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar animes
    sim_scores = sim_scores[1:6]

    # Get the anime indices
    anime_indices_recommendations = [i[0] for i in sim_scores]

    # Return the top 5 most similar animes
    return anime_df[["Name", "Japanese name", "Genres", "Score", "poster_url"]].iloc[
        anime_indices_recommendations
    ]


# Streamlit UI
st.markdown(
    '<h1 style="color: red;">AnimeStream - Anime Recommender⛩️</h1>',
    unsafe_allow_html=True,
)

st.write("##")

st.markdown(
    """
    <p>Welcome to <b style="color:#E50914;">AnimeStream</b>, your personalized anime recommendation system 
    tailored to your watch history!</p>
""",
    unsafe_allow_html=True,
)

st.write("##")

unique_user_id = users_ratings_df["user_id"].unique()
my_expander = st.expander("Tap to Select a User ID 👤💻")

selected_user_id = my_expander.selectbox("", unique_user_id)
if my_expander.button("Recommend"):
    # Fetch the animes rated by the user
    user_animes = users_ratings_df[users_ratings_df["user_id"] == int(selected_user_id)]
    user_animes = user_animes.merge(anime_df, left_on="anime_id", right_on="MAL_ID")

    if not user_animes.empty:
        # Get the most highly rated anime by the user
        top_anime = user_animes.sort_values(by="rating", ascending=False).iloc[0][
            "Name"
        ]

        # Get recommendations based on the top rated anime
        recommendations = get_recommendations(top_anime)

        st.write(
            f'Based on the top rated anime "{top_anime}", here are 5 recommendations for you:'
        )
        for index, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div style="display: flex; align-items: center;">
                    <div style="flex: 1;">
                        <img src="{row["poster_url"]}" width="150">
                    </div>
                    <div style="flex: 3;">
                        <p><b style="color:#E50914">English Name</b>: <b>{row["Name"]}</b></p>
                        <p><b style="color:#E50914">Japanese Name</b>: <b>{row["Japanese name"]}</b></p>
                        <p><b style="color:#DB4437">Rating</b>: <b>{row["Score"]}</b></p>
                        <p><b style="color:#DB4437">Genres</b>: <b>{row["Genres"]}</b></p>
                    </div>
                </div>
                <hr>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.write(f"No ratings found for User {selected_user_id}.")
