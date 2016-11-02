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
mypath = "/Users/srikanth/Downloads/Consumer_Complaints.csv"
allcomplaints=pd.read_csv(mypath)

creditcomplaints= allcomplaints[(allcomplaints.Product=="Bank account or service") &
                                (allcomplaints["Submitted via"]!="Phone" )]
# creditcomplaints=creditcomplaints[np.isfinite(creditcomplaints["Consumer complaint narrative"])]
creditcomplaints=creditcomplaints.dropna(subset=["Consumer complaint narrative"],how="all")

print creditcomplaints.head(n=5)
