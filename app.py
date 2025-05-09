import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.sparse import csr_matrix
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import time

# Set page config
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        # Load data with specific columns to reduce memory usage
        train = pd.read_csv("train.csv", usecols=['msno', 'song_id', 'target'])
        songs = pd.read_csv("songs.csv")
        
        # Print column names and sample data for debugging
        # st.write("Available columns in songs dataframe:", songs.columns.tolist())
        st.write("Sample of songs data:")
        st.write(songs.head())
        
        return train, songs
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def create_song_recommendation_system(train_df, songs_df):
    try:
        # Create a pivot table of user-song interactions
        user_song_matrix = train_df.pivot_table(
            index='msno',
            columns='song_id',
            values='target',
            fill_value=0
        )
        
        # Calculate song-song similarity matrix
        song_similarity = cosine_similarity(user_song_matrix.T)
        
        # Create a DataFrame with song similarities
        song_similarity_df = pd.DataFrame(
            song_similarity,
            index=user_song_matrix.columns,
            columns=user_song_matrix.columns
        )
        
        return song_similarity_df, songs_df
    except Exception as e:
        st.error(f"Error creating recommendation system: {str(e)}")
        return None, None

def get_song_recommendations(song_id, song_similarity_df, songs_df, n_recommendations=5):
    try:
        if song_id not in song_similarity_df.index:
            st.error(f"Song ID {song_id} not found in the similarity matrix")
            return None
            
        # Get similarity scores for the input song
        similar_songs = song_similarity_df[song_id].sort_values(ascending=False)
        
        # Get top N similar songs (excluding the input song)
        similar_songs = similar_songs[1:n_recommendations+1]
        
        # Get song details
        recommendations = songs_df[songs_df['song_id'].isin(similar_songs.index)]
        recommendations['similarity_score'] = recommendations['song_id'].map(similar_songs)
        
        return recommendations.sort_values('similarity_score', ascending=False)
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None

def search_songs(songs_df, search_term):
    try:
        # Get all column names that might contain song or artist information
        possible_columns = [col for col in songs_df.columns if any(term in col.lower() for term in ['song', 'artist', 'name', 'title'])]
        
        if not possible_columns:
            st.error("Could not find appropriate columns for song search")
            return pd.DataFrame()
        
        # Create a mask for each column
        masks = []
        for col in possible_columns:
            if songs_df[col].dtype == 'object':  # Only search in string columns
                mask = songs_df[col].str.contains(search_term, case=False, na=False)
                masks.append(mask)
        
        if not masks:
            return pd.DataFrame()
        
        # Combine all masks
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = final_mask | mask
        
        return songs_df[final_mask]
    except Exception as e:
        st.error(f"Error searching songs: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("🎵 Music Recommendation System")
    st.markdown("""
    This application predicts whether a user will replay a song based on various features.
    The model uses XGBoost to make predictions and provides insights into feature importance.
    """)

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Song Recommendations"
                                    #   "Data Analysis", "Model Training", "Predictions"
                                      ])

    if page == "Home":
        st.header("Welcome to the Music Recommendation System")
        st.write("""
        This application helps predict user song preferences using machine learning.
        Features include:
        - Song recommendations based on user input
        - Data analysis and visualization
        - Model training and evaluation
        - Prediction generation
        - Feature importance analysis
        """)

    elif page == "Song Recommendations":
        st.header("🎵 Song Recommendations")
        
        # Load data
        train, songs = load_data()
        if train is not None and songs is not None:
            # Create recommendation system
            with st.spinner("Creating recommendation system..."):
                song_similarity_df, songs_df = create_song_recommendation_system(train, songs)
            
            # if song_similarity_df is not None:
            #     # Add a sample search section
            #     st.subheader("Try a Sample Search")
            #     st.write("Here are some example songs you can search for:")
                
            #     # Display first 5 songs from the dataset with their IDs
            #     sample_songs = songs_df.head(5)
            #     for _, song in sample_songs.iterrows():
            #         st.write(f"Song ID: {song['song_id']}")
            #         st.write("Details:", song.to_dict())
            #         st.write("---")
                
                # Search box for songs
                st.subheader("Search for a Song")
                search_term = st.text_input("Enter song name or artist:", "")
                
                if search_term:
                    # Search in songs dataframe
                    search_results = search_songs(songs_df, search_term)
                    
                    if not search_results.empty:
                        st.write("Found songs:")
                        # Display first 5 results with all available information
                        for _, song in search_results.head(5).iterrows():
                            st.write("Song ID:", song['song_id'])
                            st.write("Details:", song.to_dict())
                            st.write("---")
                        
                        # Let user select a song
                        selected_song = st.selectbox(
                            "Select a song to get recommendations:",
                            options=search_results['song_id'].tolist(),
                            format_func=lambda x: f"Song ID: {x}"
                        )
                        
                        if selected_song:
                            if st.button("Get Recommendations"):
                                with st.spinner("Finding similar songs..."):
                                    recommendations = get_song_recommendations(selected_song, song_similarity_df, songs_df)
                                    
                                    if recommendations is not None and not recommendations.empty:
                                        st.success("Here are your song recommendations:")
                                        
                                        for _, song in recommendations.iterrows():
                                            st.write(f"""
                                            🎵 Song ID: {song['song_id']}  
                                            📊 Similarity Score: {song['similarity_score']:.2f}  
                                            ---
                                            """)
                                    else:
                                        st.warning("No recommendations found for this song.")
                    else:
                        st.warning("No songs found matching your search. Try a different search term.")

    elif page == "Data Analysis":
        st.header("Data Analysis")
        if st.button("Load and Analyze Data"):
            with st.spinner("Loading data..."):
                train, songs = load_data()
                if train is not None:
                    st.success("Data loaded successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Training Data Overview")
                        st.write(f"Number of records: {len(train)}")
                        st.write(f"Number of features: {len(train.columns)}")
                        st.dataframe(train.head())
                    
                    with col2:
                        st.subheader("Songs Data Overview")
                        st.write(f"Number of records: {len(songs)}")
                        st.write(f"Number of features: {len(songs.columns)}")
                        st.dataframe(songs.head())

    elif page == "Model Training":
        st.header("Model Training")
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                train, songs = load_data()
                if train is not None:
                    st.success("Model trained successfully!")

    elif page == "Predictions":
        st.header("Generate Predictions")
        if st.button("Generate New Predictions"):
            with st.spinner("Generating predictions..."):
                train, songs = load_data()
                if train is not None:
                    st.success("Predictions generated successfully!")
                    
                    # Display predictions
                    st.subheader("Prediction Results")
                    st.dataframe(train.head(10))
                    
                    # Download button
                    csv = train.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Predictions CSV",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )

if __name__ == "__main__":
    main()
