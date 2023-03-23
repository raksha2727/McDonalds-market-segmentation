#!/usr/bin/env python
# coding: utf-8
Importing libaries
# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().system('pip install bioinfokit')


# In[23]:


#data visualization
import matplotlib.pyplot as plt
import seaborn as sns 

#clustering model library
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

DATA GATHERING
# In[3]:


#Load the datset using pandas
df = pd.read_csv('mcdonalds.csv')
df.head(10)


# In[4]:


#Loading the dataset
df.shape
df.head()
df.dtypes
# 11 variable(cols) has yes or no values.

# checking for null data --> No null data
df.info()
df.isnull().sum()


# In[6]:


#shape of dataset
df.shape


# In[7]:


#Data types of dataframe
df.dtypes


# In[8]:


df.info()


# In[9]:


df.describe()


# In[ ]:


five number summary is available for age because age is only integer field

DATA PREEPROCESSING
# In[10]:


#Checking the null value
df.isnull().sum()


# In[ ]:


so , no null value is present in dataset


# Checking count for important fields
df['Gender'].value_counts()
# In[12]:


df['VisitFrequency'].value_counts()


# In[13]:


df['Like'].value_counts()


# In[14]:


#label replace perform label encoding
df['Like'].replace('I hate it!-5',-5,inplace = True)
df['Like'].replace('I love it!+5', 5,inplace = True)


# #EXPLORATORY DATA ANALYSIS

# In[15]:


# Distribution using gender
labels = ['Female', 'Male']
size = df['Gender'].value_counts()
colors = ['pink', 'blue']
explode = [0, 0.1]
plt.rcParams['figure.figsize'] = (6, 6)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Gender', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()


# In[18]:


#distribution on age
import seaborn as sns
plt.rcParams['figure.figsize'] = (25, 8)
f = sns.countplot(x=df['Age'],palette = 'hsv')
#f.bar_label(f.containers[0])
plt.title('Age distribution of customers')
plt.show()

We can easily see distribution of age between 18 and 70 is formed like uniform distribution not 
exactly but look likeCustomer segmentation - based on pyschographic segmentation
# In[17]:


# count Rating with respect to age
df['Like']= df['Like'].replace({'I hate it!-5': '-5','I love it!+5':'+5'})
#Like 
sns.catplot(x="Like", y="Age",data=df, 
            orient="v", height=5, aspect=2, palette="Set2",kind="swarm")
plt.title('Likelyness of McDonald w.r.t Age')
plt.show()

Convert all the categorical varable to numerical variable using labelencoder
# In[20]:


#Label encoding for categorical - Converting 11 cols with yes/no

from sklearn.preprocessing import LabelEncoder
def labelling(x):
    df[x] = LabelEncoder().fit_transform(df[x])
    return df

cat = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap',
       'tasty', 'expensive', 'healthy', 'disgusting']

for i in cat:
    labelling(i)
df

0 - no¶
1 - yesone thing that is important in this dataset all the field is import and relevant to get the
conclusion of segmenatation so we can not neglect any field of dataset
# In[27]:


#Histogram of the each attributes
plt.rcParams['figure.figsize'] = (12,14)
df.hist()
#plt.show()

# considering first 11 fields of dataset so we find the principal axis it helful to
reduce the dimension because here all the field is important
# In[28]:


df_imp = df.loc[:,cat]
df_imp

converting these 11 colomns into arrays
# In[29]:


x = df.loc[:,cat].values
x

create principal component axis along these colomns
# In[30]:


from sklearn.decomposition import PCA
from sklearn import preprocessing

pca_data = preprocessing.scale(x)

pca = PCA(n_components=11)
pc = pca.fit_transform(x)
names = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11']
pf = pd.DataFrame(data = pc, columns = names)
pf


# In[31]:


#Proportion of Variance (from PC1 to PC11)
pd.Series(pca.explained_variance_ratio_)

so after finding the PCA axis we can see last three PCA axis does not give more variance
# In[32]:


np.cumsum(pca.explained_variance_ratio_)

first seven PCA axis are capable to give 85% of information

Now we find Corelation between original component and PCA components
# In[33]:


loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = df_imp.columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df


# In[37]:


from bioinfokit.visuz import cluster
cluster.screeplot(obj=[pc_list, pca.explained_variance_ratio_],show=True,dim=(10,5))

Corelation matric for above dataset
# In[45]:


plt.figure(figsize=(20,15))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[42]:


import math
import seaborn as sns
import os
import matplotlib.pyplot as plt


# In[51]:


#Correlation matrix plot for loadings 
plt.rcParams['figure.figsize'] = (20,15)
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()

PC1 has large variance
# In[129]:



pca_scores = PCA().fit_transform(x)

cluster.biplot(cscore=pca_scores, loadings=loadings, labels=df.columns.values, var1=round(pca.explained_variance_ratio_[0]*100, 2),
    var2=round(pca.explained_variance_ratio_[1]*100, 2),show=True,dim=(10,5))

PC1 and PC2 ahs large variance where maximum number of point lie on the axis that's why this
is the more important axis for us

Now we use k-means clustring algorithm to formed a cluster for same time of customer so each
cluster is formed on the basis of distortin score elbow method so we can easily decide how to
find the optimer number of cluster based on the dissimilarity between user using elbow method .
we use different value of K and whichever value of K Distortion score is high we pick that one
according to value of K we decide number of cluster
# EXTRACTING SEGMENTS

# In[56]:


get_ipython().system('pip install yellowbrick')


# In[64]:


get_ipython().system('pip install yellowbrick')

K-means Clustring
# In[66]:


#Using k-means clustering analysis
from sklearn.cluster import KMeans
#from yellowbrick.cluster import KElbowVisualizer
#model = KMeans()
#visualizer = KElbowVisualizer(model, k=(1,12)).fit(df_imp)
#visualizer.show()
#plt.show()


# In[69]:


kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df_imp)
df['cluster_num'] = kmeans.labels_ #adding to df
print (kmeans.labels_) #Label assigned for each data point
print (kmeans.inertia_) #gives within-cluster sum of squares. 
print(kmeans.n_iter_) #number of iterations that k-means algorithm runs to get a minimum within-cluster sum of squares
print(kmeans.cluster_centers_) #Location of the centroids on each cluster.

Each cluster Size
# In[70]:


from collections import Counter
Counter(kmeans.labels_)


# In[ ]:


Cluster Visualization


# In[89]:


#Visulazing clusters
sns.scatterplot(data=pf, x="pc1", y="pc2", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()

Audience in each cluster based on gender
# In[ ]:


DESCRIBING SEGMENTS


# In[77]:


crosstab_gender =pd.crosstab(df['cluster_num'],df['Gender'])
crosstab_gender


# In[92]:


plt.rcParams['figure.figsize'] = (7,5)
mosaic(crosstab_gender.stack())
plt.show()

Audience in each cluster based on Age
# In[79]:


sns.boxplot(x="cluster_num", y="Age", data=df)


# In[ ]:


mean of Visit Frequency of each Cluster


# In[ ]:


Selecting target segment


# In[80]:


#Calculating the mean
#Visit frequency
df['VisitFrequency'] = LabelEncoder().fit_transform(df['VisitFrequency'])
visit_freq = df.groupby('cluster_num')['VisitFrequency'].mean()
visit_freq = visit_freq.to_frame().reset_index()
visit_freq

Highest Visit frequency of auidience in cluster 3

vISIT fREQUENCY RANGE(2.54 TO 2.65)
# In[81]:


df['Like'] = df['Like'].astype('int')


# In[82]:


df['Like'] = LabelEncoder().fit_transform(df['Like'])
Like = df.groupby('cluster_num')['Like'].mean()
Like = Like.to_frame().reset_index()
Like


# In[83]:


df['Like'].value_counts()


# In[132]:


segment = Gender.merge(Like, on='cluster_num', how='left').merge(visit_freq, on='cluster_num', how='left')
segment


# In[107]:


segment = crosstab_gender.merge(Like, on='cluster_num', how='left').merge(visit_freq, on='cluster_num', how='left')
segment


# In[133]:


plt.figure(figsize = (9,4))
sns.scatterplot(x = "VisitFrequency", y = "Like",data=segment,s=400, color="r")
plt.title("Simple segment evaluation plot for the fast food data set",
          fontsize = 15) 
plt.xlabel("Visit", fontsize = 12) 
plt.ylabel("Like", fontsize = 12) 
plt.show()


# In[134]:


Gender


# In[135]:


plt.figure(figsize = (9,4))
sns.scatterplot(x = "VisitFrequency", y = "Gender",data=segment,s=400, color="r")
plt.title("Simple segment evaluation plot for the fast food data set",
          fontsize = 15) 
plt.xlabel("Visit", fontsize = 12) 
plt.ylabel("Gender", fontsize = 12) 
plt.show()


# In[117]:


#Target segments

plt.figure(figsize = (9,4))
sns.scatterplot(x = "VisitFrequency", y = "Like",data=segment,s=400, color="r")
plt.title("Simple segment evaluation plot for the fast food data set",
          fontsize = 15) 
plt.xlabel("Visit", fontsize = 12) 
plt.ylabel("Like", fontsize = 12) 
plt.show()

NOw The conclusion of segmentation is based on the Visitfrequency and Like we can select our cluster and according to cluster audience we target that audince like age and gender also taste is involve for target the audience . The target audienced that is highly focused for us is that visitfrequency mean between 2.50 to 2.60 followed by other AgglomerativeClustering
# In[155]:


import scipy.cluster.hierarchy as sch
plt.figure(figsize=(10, 7))


dendrogram = sch.dendrogram(sch.linkage(df, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('Gender')
plt.ylabel('Visit')
plt.show()


# In[156]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cl = cluster.fit_predict(df)
cl


# In[158]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import silhouette_score


# In[159]:


silhouette_score(df,cl)


# In[161]:


sc_list=[]

for i in range(2,6):
    cluster = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')  
    cls = cluster.fit_predict(df)
    sc = silhouette_score(df,cls)
    sc_list.append(sc)
    
    print(f'value of k is {i} & value of seilhoutte score is {sc}')


# In[162]:


sc_list


# In[163]:


X = df.values


# In[164]:


X


# In[166]:


plt.figure(figsize=(10, 7))  
plt.scatter(X[cl==0, 0], X[cl==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[cl==1, 0], X[cl==1, 1], s=100, c='blue', label ='Cluster 2')
plt.title('Customer segmentation')
plt.xlabel('Visit')
plt.ylabel('Gender')
plt.show()

Refrences
# https://ai-pool.com/a/s/finding-an-optimal-number-of-clusters-with-elbow-method https://www.youtube.com/watch?v=1XqG0kaJVHY
# 
# https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-learning
# 
# https://rpubs.com/tmk1221/segmentation

# In[169]:


#Target segments

plt.figure(figsize = (9,4))
sns.scatterplot(x = "VisitFrequency", y = "Like",data=segment,s=400, color="r")
plt.title("Simple segment evaluation plot for the fast food data set",
          fontsize = 15) 
plt.xlabel("Visit", fontsize = 12) 
plt.ylabel("Like", fontsize = 12) 
plt.show()


# In[ ]:




