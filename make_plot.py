import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import re
from sklearn import mixture

file_name_kmeans = 'output/face_clusters_AU12_r_2.csv' 
file_name_gmm = 'output/gmm_AU12_r_3.csv'

df = pd.read_csv(file_name_kmeans)
cluster_data = df.values

df_gmm = pd.read_csv(file_name_gmm)
mean = df_gmm['AU12_r'].values
sigma = df_gmm['sigmas'].values

df1 = pd.read_pickle('all_frames.pkl.xz')
data = df1['AU12_r'].values

num_clusters = cluster_data.shape[0]
n = data.shape[0]

data = data.reshape(n , 1)
num_bins = 100

#plot k-means
plt.figure(figsize= (8,8))
plt.subplot(211)
plt.hist(data, bins=num_bins, color='green', histtype='bar', \
                 ec='black', normed=True)
plt.scatter(cluster_data[:,0], np.zeros(num_clusters), s=500, \
                        c='red')
plt.xlabel('AU12 intensity')
plt.ylabel('Nornamlized frequency')

#plot gmm
plt.subplot(212)
x = np.linspace(min(data), max(data), 1000) 
plt.hist(data, bins=num_bins, color='green', histtype='bar', \
         ec='black', normed=True)
plt.xlabel('AU12 intensity')
plt.ylabel('Nornamlized frequency')

for i in range(0, mean.shape[0]):
    current_sigma = float(re.findall("\d+\.\d+", sigma[i])[0])
    current_mean = mean[i]
    rv = multivariate_normal(mean = current_mean, cov = current_sigma)
    pdf_array = rv.pdf(x)
    #plt.scatter(x, pdf_array, s = 1)
    pd.Series(pdf_array, x).plot()
    
    



plt.show()




