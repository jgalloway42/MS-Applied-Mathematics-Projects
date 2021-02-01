from IPython.display import set_matplotlib_formats, display
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons, make_blobs
from itertools import combinations

set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['image.cmap'] = "autumn_r"
plt.rcParams['image.interpolation'] = "none"
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['figure.facecolor'] = '#d3d3d3'
plt.rcParams['figure.autolayout'] = 'False'

np.set_printoptions(precision=3, suppress=True)

pd.set_option("display.max_columns", 15)
pd.set_option('precision', 2)