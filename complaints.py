from cStringIO import StringIO
import os
import re
import math
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from nltk import bigrams
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np

#depfining the path where all the files are stored
mypath = "./Data/Consumer_Complaints.csv"
allcomplaints=pd.read_csv(mypath)

creditcomplaints= allcomplaints[(allcomplaints.Product=="Credit card") &
                                (allcomplaints["Submitted via"]!="Phone" )]
# creditcomplaints=creditcomplaints[np.isfinite(creditcomplaints["Consumer complaint narrative"])]
creditcomplaints=creditcomplaints.dropna(subset=["Consumer complaint narrative"],how="all")
#implementing Latent drichilet allocation
tokenizer = RegexpTokenizer(r'\w+')

#create English stop words list
en_stop = get_stop_words('en')

#including domain specific stop words
my_stopwords = ["xx","xxxx"]
my_stopwords1= [i.decode('utf-8') for i in my_stopwords]
en_stop = en_stop +my_stopwords1

texts=[]
complaintsnarrative = creditcomplaints["Consumer complaint narrative"].tolist()
#Loop through the documents
for i in complaintsnarrative:
    #clean and tokenize document string
    #temp = i
    raw = i.lower()
    #stem tokens
    #stemmed_tokens = [p_stemmer.stem(i) for i in tokenizer.tokenize(raw)]
    #remove stop words from tokens

    #split data, remove stop words and convert to bigrams
    tokens = bigrams(i for i in tokenizer.tokenize(raw)if not i in en_stop and len(i)>1)

    #stem tokens
    #stemmed_tokens = [p_stemmer.stem(i) for i in tokens]
    #remove stop words from tokens

    mergedTokens = [i[0]+" "+i[1] for i in tokens]
    stopped_tokens = [i for i in mergedTokens if not i in en_stop]

    #add tokens to list
    texts.append(stopped_tokens)

#turn our tokenized documents into # a documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

#convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.LdaModel(corpus, num_topics = 5 , id2word = dictionary, passes = 1)
print(ldamodel.show_topics(num_topics=5))