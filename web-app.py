import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import nmslib
import pickle
import scipy.sparse as sparse
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

@st.cache
def read_files(folder_name):
    ratings = pd.read_csv(folder_name + '/ratings.csv')
    books = pd.read_csv(folder_name + '/books.csv')
    return ratings, books 

def make_mappers(books):
    name_mapper = dict(zip(books.book_id, books.title))
    author_mapper = dict(zip(books.book_id, books.authors))

    return name_mapper, author_mapper

def load_embeddings(file_name):
    with open(file_name, 'rb') as f:
        item_embeddings = pickle.load(f)
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings, nms_idx

def nearest_books_nms(book_id, index, n=10):
    nn = index.knnQuery(item_embeddings[book_id], k=n)
    return nn

def get_recomendation_df(ids, distances, name_mapper, author_mapper):
    names = []
    authors = []
    for idx in ids:
        names.append(name_mapper[idx])
        authors.append(author_mapper[idx])
    recomendation_df = pd.DataFrame({'book_name': names, 'book_author': authors, 'distance': distances})
    return recomendation_df

ratings, books = read_files('tables') 
name_mapper, author_mapper = make_mappers(books)
item_embeddings, nms_idx = load_embeddings('books-embeddings.pkl')


st.title("Рекомендации книг")

st.markdown("""

Для получения списка рекомендаций:
1. Введите приблизительное название книги
2. Выберите точное соответствие
3. Выберите количество книг для рекомендаций

""")

title = st.text_input('Введите название книги для быстрого поиска')
title = title.lower()

output = books[books['title'].apply(lambda x: x.lower().find(title)) >= 0]

option = st.selectbox("Выберите книгу для подготовки рекомендаций", output['title'].values)

if option:
    st.markdown('Выбрано: "{}"'.format(option))
    val_index = output[output['title'].values == option]['book_id'].values
    count_recomendation = st.number_input(
        label="Выберите сколько рекомендацией необходимо подобрать", 
        value=5
    )    
    ids, distances = nearest_books_nms(val_index, nms_idx, count_recomendation+1)
    ids, distances = ids[1:], distances[1:]
    st.markdown('Рекомендуемые книги: ')
    df = get_recomendation_df(ids, distances, name_mapper, author_mapper)
    st.dataframe(df[['book_name', 'book_author']])
    fig = px.bar(
        data_frame=df, 
        x='book_name', 
        y='distance',
        hover_data=['book_author'],
        title='Рекомендации'
    )
    st.write(fig)
