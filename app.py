import streamlit as st
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import FlagReranker
import pandas as pd
import numpy as np

@st.cache_resource
def load_model():
    return BGEM3FlagModel('BAAI/bge-m3',
                        use_fp16=True)
@st.cache_resource
def load_reranker():
    return FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

@st.cache_data
def load_df(path):
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_embed(path):
    embeddings_2 = np.load(path)
    return embeddings_2

model = load_model()
reranker = load_reranker()

df = load_df('BookDataFrame.csv')
embeddings_2 = load_embed('BGE_embeddings_2.npy')

st.header(":books: Book Identifier")

k = 10
with st.form(key='my_form'):
	sen1 = st.text_area("Book description:")
	submit_button = st.form_submit_button(label='Submit')

if submit_button:
    embeddings_1 = model.encode(sen1,
                                batch_size=12,
                                max_length=8192,
                                )['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T

    top_k_qs = []
    topk = np.argsort(similarity)[-k:]

    for t in topk:
        pred_sum = df['Summary'].iloc[t]
        pred_ques = sen1
        pred = [pred_ques, pred_sum]
        top_k_qs.append(pred)
    rrscore = reranker.compute_score(top_k_qs, normalize=True)
    rrscore_index = np.argsort(rrscore)

    pred_book = []
    for rr in rrscore_index:
        pred_book.append(f"{df['Book Name'][topk[rr]]} by {df['Book Author'][topk[rr]]}")

    finalpred = []
    pred_book.reverse()
    st.write("Here is your prediction")
    for n, pred in enumerate(pred_book):
        st.write(f"{n+1}: {pred}")
