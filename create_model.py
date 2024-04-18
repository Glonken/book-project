import numpy as np
import pandas as pd
import scipy.sparse as sparse
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, recall_at_k
import pickle, nmslib

def nearest_books_nms(book_id, index, n=10):
    nn = index.knnQuery(item_embeddings[book_id], k=n)
    return nn

ratings = pd.read_csv('tables/ratings.csv')
books = pd.read_csv('tables/books.csv')
tags = pd.read_csv('tables/tags.csv')
book_tags = pd.read_csv('tables/tag-books.csv')

books.head()

mapper = dict(zip(books.goodreads_book_id,books.book_id))

tags = pd.read_csv('tables/clear-tags.csv')
book_tags = book_tags[book_tags.tag_id.isin(tags.tag_id)]
book_tags['id'] = book_tags.goodreads_book_id.apply(lambda x: mapper[x])

ratings_coo = sparse.coo_matrix((ratings.rating,(ratings.user_id, ratings.book_id)))
feature_ratings  = sparse.coo_matrix(([1]*len(book_tags), (book_tags.id, book_tags.tag_id)))

NUM_THREADS = 10
NUM_COMPONENTS = 60
NUM_EPOCHS = 15
RANDOM_STATE = 42

train, test = random_train_test_split(ratings_coo, test_percentage=0.2, random_state=RANDOM_STATE)
model = LightFM(learning_rate=0.05, loss='warp', no_components=NUM_COMPONENTS, random_state=RANDOM_STATE)
model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS, item_features=feature_ratings)

precision_score = precision_at_k(model, test, num_threads=NUM_THREADS, k=10, item_features=feature_ratings).mean()
recall_score = recall_at_k(model, test, num_threads=NUM_THREADS, k=10, item_features=feature_ratings).mean()

print(recall_score, precision_score)

with open('book-model.pkl', 'wb') as file:
    pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

with open('book-model.pkl', 'rb') as file:
    model = pickle.load(file)

item_biases, item_embeddings = model.get_item_representations(features=feature_ratings)

print(item_biases.shape, item_embeddings.shape)

nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
nms_idx.addDataPointBatch(item_embeddings)
nms_idx.createIndex(print_progress=True)

books[books['title'].apply(lambda x: x.lower().find('1984')) >= 0]

print(nearest_books_nms(846, nms_idx))

nbm = nearest_books_nms(846, nms_idx)[0]
books[books.book_id.isin(nbm)][['authors', 'title']]

with open('books-embeddings.pkl', 'wb') as file:
    pickle.dump(item_embeddings, file, protocol=pickle.HIGHEST_PROTOCOL)
