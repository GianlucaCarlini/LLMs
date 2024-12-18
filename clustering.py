from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.document_loaders import PyPDFLoader
from langchain import text_splitter as ts
import pandas as pd
import glob
import os
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

__all__ = ["cluster_documents", "label_clusters", "plot_clusters"]


def cluster_documents(folder, chunk_size=6000):

    clustering_progress_bar = st.progress(0)

    docs = glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True)

    embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

    text_splitter = ts.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0
    )

    embedded_docs = {}

    for i, doc in enumerate(docs):

        try:
            loader = PyPDFLoader(doc)

            doc_text = loader.load()

            all_splits = text_splitter.split_documents(doc_text)

            # take the first split only
            doc_text = all_splits[0].page_content

            embedded_docs[doc] = embedding_model.embed_query(doc_text)

        except Exception as e:
            print(f"Error while processing {doc}")

        clustering_progress_bar.progress((i + 1) / len(docs))

    embedded_docs_df = pd.DataFrame.from_dict(embedded_docs, orient="index")

    umap_modl = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    )

    umap_embeddings = umap_modl.fit_transform(embedded_docs_df)

    umap_df = pd.DataFrame(
        umap_embeddings, index=embedded_docs_df.index, columns=["x", "y"]
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=5,
        cluster_selection_epsilon=0.5,
        metric="euclidean",
    )

    umap_df["cluster"] = clusterer.fit_predict(umap_embeddings)

    return umap_df


def label_clusters(df, chunk_size=6000):

    label_progress_bar = st.progress(0)

    summarization_model = ChatOllama(model="llama3.2:3b-instruct-fp16", temperature=0)

    text_splitter = ts.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0
    )

    clustered_docs = {}

    for i, row in enumerate(df.iterrows()):
        loader = PyPDFLoader(row[0])

        doc_text = loader.load()

        all_splits = text_splitter.split_documents(doc_text)

        # take the first split only
        doc_text = all_splits[0]

        # use the cluster number as key for the dictionary
        cluster = row[1]["cluster"]

        if cluster not in clustered_docs:
            clustered_docs[cluster] = []

        clustered_docs[cluster].append(doc_text)
        label_progress_bar.progress((i + 1) / len(df))

    label_progress_bar = st.progress(0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """Use one single keyword to describe the content of the following document: \\n \\n {context} \n
        Do not provide a summary of the document, just one keyword which is representative of the content""",
            )
        ]
    )

    chain = create_stuff_documents_chain(summarization_model, prompt)

    keywords = {}

    i = 0

    for key, value in tqdm(clustered_docs.items(), desc="Summarizing clusters"):

        docs = value

        result = chain.invoke({"context": docs})

        print(result)

        keywords[key] = result

        i += 1

        label_progress_bar.progress(i / len(clustered_docs))

    df["keyword"] = df["cluster"].map(keywords)

    return df


def plot_clusters(df):

    fig = plt.figure(figsize=(10, 10))

    sns.scatterplot(data=df, x="x", y="y", hue="keyword", palette="tab10")

    return fig
