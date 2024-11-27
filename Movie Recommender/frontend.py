import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import requests



def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=2a2b5738f05b199fd1a64e4fe7aee9c2&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    similarity = distance[movie_index]
    movies_list = sorted(list(enumerate(similarity)), reverse = True, key = lambda x:x[1])[1:6]
    
    recommended_movies = []
    rec_movie_posters = []
    for i in movies_list:
        mov_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        #fetch poster API
        rec_movie_posters.append(fetch_poster(mov_id))

    return recommended_movies, rec_movie_posters

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

distance = pickle.load(open('distance.pkl', 'rb'))

selected_movie_name = st.selectbox(
"In the meantime, pick a movie.", 
movies['title'].values)

if st.button('Recommend'):
    recommendations,posters= recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommendations[0])
        st.image(posters[0])

    with col2:
        st.text(recommendations[1])
        st.image(posters[1])

    with col3:
        st.text(recommendations[2])
        st.image(posters[2])
    with col4:
        st.text(recommendations[3])
        st.image(posters[3])
    with col5:
        st.text(recommendations[4])
        st.image(posters[4])


page_bg_img = """
<style>
[data-testid = "stAppViewContainer"]{
background-image: url("https://www.pixelstalk.net/wp-content/uploads/images6/Vaporwave-Wallpaper-High-Quality.png");
background-size: cover;

}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
#st.Title("Chic Flicks")