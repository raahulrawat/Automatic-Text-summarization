# Textsum

# Automatic Text Summarization / Key Sentences Extraction

TextRank Algorithm

Let’s understand the TextRank algorithm, now that we have a grasp on PageRank. I have listed the similarities between these two algorithms below:

    In place of web pages, we use sentences
    Similarity between any two sentences is used as an equivalent to the web page transition probability
    The similarity scores are stored in a square matrix, similar to the matrix M used for PageRank

TextRank is an extractive and unsupervised text summarization technique. Let’s take a look at the flow of the TextRank algorithm that we will be following:

    1. The first step would be to concatenate all the text contained in the articles
    2. Then split the text into individual sentences
    3. In the next step, we will find vector representation (word embeddings) for each and every sentence
    4. Similarities between sentence vectors are then calculated and stored in a matrix
    5. The similarity matrix is then converted into a graph, with sentences as vertices and similarity scores as edges, for sentence rank calculation
    6. Finally, a certain number of top-ranked sentences form the final summary


Testing Dataset link: https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/tennis_articles_v4.csv


Download GloVe Word Embeddings

GloVe word embeddings are vector representation of words. These word embeddings will be used to create vectors for our sentences. We could have also used the Bag-of-Words or TF-IDF approaches to create features for our sentences, but these methods ignore the order of the words (and the number of features is usually pretty large).

We will be using the pre-trained Wikipedia 2014 + Gigaword 5 GloVe vectors available here. Heads up – the size of these word embeddings is 822 MB.

Commands:
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip


Requirements:
numpy
pandas
nltk
re
networkx
sklearn

nltk data: "punkt", "stopwords"


Running code:

give text input to the method summary() and this function will return the summarization of the given input text.

example:
from textsum import Summarize

#loading Summarize model
summarize = Summarize()

input_text = "Advances in the classification of individual cooking ingredients are sparse. The problem is that there are almost no public edited records available. This work deals with the problem of automated recognition of a photographed cooking dish and the subsequent output of the appropriate recipe. The distinction between the difficulty of the chosen problem and previous supervised classification problems is that there are large overlaps in food dishes (aka high intra-class similarity), as dishes of different categories may look very similar only in terms of image information."

summary = summarize.summary(text = input_text)

print(summary)

Advances in the classification of individual cooking ingredients are sparse. The problem is that there are almost no public edited records available. This work deals with the problem of automated recognition of a photographed cooking dish and the subsequent output of the appropriate recipe. The distinction between the difficulty of the chosen problem and previous supervised classification problems is that there are large overlaps in food dishes (aka high intra-class similarity), as dishes of different categories may look very similar only in terms of image information.


