import streamlit as st
import pickle as pk
import pandas as pd
st.set_page_config(page_title='Book Recommender', page_icon='📚', layout='centered', initial_sidebar_state='auto')
st.title('Book Recommender')

books_names = list(pk.load(open('./models/book_names.pkl', 'rb')))
books_pivot = pd.read_pickle(open('./models/book_pivot.pkl', 'rb'))
book_metadata = pd.read_pickle('./models/book_metadata.pkl')
knn_model = pk.load(open('./models/model.pkl', 'rb'))

def display_metadata(book_name):
    book = book_metadata[book_metadata['Book-Title'] == book_name].sample()
    # Create a two-column layout
    col1, col2 = st.columns([3,1])  # Ratio controls the width of the columns
    
    # Display text data on the left (col1)
    with col1:
        st.markdown(f"**Title**: {book['Book-Title'].values[0]}")
        st.markdown(f"**ISBN**: {book['ISBN'].values[0]}")
        st.markdown(f"**Author**: {book['Book-Author'].values[0]}")
        st.markdown(f"**Year**: {book['Year-Of-Publication'].values[0]}")
    # Display image on the right (col2)
    with col2:
        st.image(book['Image-URL-L'].values[0], width=200)  # Adjust image size as needed

def recommend(book_name):
    book_index = books_names.index(book_name)
    distances, indices = knn_model.kneighbors(books_pivot.iloc[book_index, :].values.reshape(1, -1), n_neighbors=6)
    recommendations = []
    for i in indices.flatten():
        recommendations.append(books_names[i])
    return recommendations[1:]

book_name = st.selectbox('Choose a Book', books_names)
btn = st.button('Select')
if  btn and book_name is not None:
    display_metadata(book_name)
    st.divider()
    st.markdown("<b>Recommended for you:</b>", unsafe_allow_html=True)
    st.write(recommend(book_name))
