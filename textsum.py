import numpy as np
import pandas as pd
import nltk
import re
import networkx as nx
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from sklearn.metrics.pairwise import cosine_similarity


class Summarize:

    def __init__(self):

        # define glove model path here !!
        self.glove_model = "glove.6B.100d.txt"

    # function to remove stopwords
    def remove_stopwords(self, sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    def summary(self, text):

        data = text
        sentences = []
        sentences.append(sent_tokenize(data))
        sentences = [y for x in sentences for y in x]  # flatten list

        # Extract word vectors
        word_embeddings = {}
        f = open(self.glove_model, encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
        f.close()

        # remove punctuations, numbers and special characters
        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

        # make alphabets lowercase
        clean_sentences = [s.lower() for s in clean_sentences]

        # remove stopwords from the sentences
        clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

        sentence_vectors = []
        for i in clean_sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)

        # similarity matrix
        sim_mat = np.zeros([len(sentences), len(sentences)])

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        # Extract top 4 sentences as the summary
        for i in range(4):
            print(ranked_sentences[i][1])
