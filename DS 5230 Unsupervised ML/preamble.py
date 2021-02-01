'''Imports'''
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits import mplot3d
import seaborn as sns
import os 
import pickle
from datetime import datetime
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentAnalyzer,vader
from string import punctuation,digits
from time import time
from sklearn.decomposition import LatentDirichletAllocation,NMF,TruncatedSVD,PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')



'''Constants'''
COLORS = ['tab:blue', 'tab:orange', 'tab:green',
       'tab:red', 'tab:purple', 'tab:brown',
       'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']