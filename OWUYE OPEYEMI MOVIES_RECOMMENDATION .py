#!/usr/bin/env python
# coding: utf-8

# __CREATING A RECOMMENDATION SYSTEM WITH COSINE SIMILARITY__

# __DATASET FROM KAGGLE: https://www.kaggle.com/datasets/gan2gan/1000-imdb-movies-20062016 and can be accessed here__

# In[1]:


#importing our libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('IMDB-Movie-Data.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df=df.dropna()
df.isnull().sum()
#drop null values


# In[6]:


#Cheching for Duplicated values
df.duplicated().sum()


# In[7]:


df.reset_index(inplace=True, drop=True)


# In[8]:


df.drop(['Rank'], axis=1, inplace=True)
df.head()


# __Exploratory Data Analysis__

# In[9]:


#EDA
#1. what year had the highest movie produced
sns.countplot(data=df, x="Year")


# __IN 2016, The highest number of movies was produced__

# In[10]:


plt.figure(figsize=(14, 8))
sns.countplot(data=df, x="Rating");


# __FEATURE ENGINEERING__

# In[11]:


#COMBINING OUR TEXT DATA
df['text']= df['Title'] + " " + df['Genre'] + " " + df['Description'] + " " + df['Director'] + " " + df['Actors']
df.head()


# In[12]:


Movies=df.iloc[:, [0,1,2,-1]]
Movies.head() #extracting the dataset we want to work with


# In[13]:


Movies.info()


# In[14]:


Movies["text"][0]


# In[15]:


Movies['ids']=[i for i in range(0, Movies.shape[0])] #creating an id column


# In[16]:


Movies.head()


# __TEXT PREPROCESSING__

# In[17]:


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# In[18]:


import nltk
nltk.download('stopwords')


# In[19]:


stopwords = nltk.corpus.stopwords.words('english')


# In[20]:


lemmatizer = WordNetLemmatizer()


# In[21]:


nltk.download('punkt')


# In[22]:


def clean(text):
    text = re.sub("[^A-Za-z1-9 ]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    clean_list = []
    for token in tokens:
        if token not in stopwords:
            clean_list.append(lemmatizer.lemmatize(token))
    return " ".join(clean_list)


# In[23]:


test_text=Movies.text[0]
test_text


# In[24]:


import nltk
nltk.download('omw-1.4')


# In[25]:


clean(test_text)


# In[26]:


Movies.text=Movies.text.apply(clean)


# In[27]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
CV = CountVectorizer()
converted_metrix = CV.fit_transform(df.text)


# In[28]:


cosine_similarity = cosine_similarity(converted_metrix)


# In[29]:


cosine_similarity


# In[30]:


#Testing 
Movies[Movies['text'] .str.contains('action')]


# In[31]:


Movies["Genre"][0]


# In[32]:


Genre = 'Action,Adventure,Sci-Fi'
genre_id=Movies[Movies["Genre"]==Genre]["ids"].values[0]


# In[33]:


similar_genre = list(enumerate(cosine_similarity[genre_id]))


# In[34]:


sorted_similar_genre=sorted(similar_genre, key=lambda x:x[1], reverse= True)
sorted_similar_genre=sorted_similar_genre[1: ]


# In[35]:


sorted_similar_genre[0:10]


# In[36]:


i =0
for item in sorted_similar_genre:
    movie_genre= Movies[Movies["ids"] ==item[0]]["Genre"].values[0]
    print(i + 1, movie_genre)
    i = i+1
    if 1> 5:
        break


# __BUILDING OUR RECOMMENDATION SYSTEM__
# 
# __BASED ON MOVIE TITLE__

# In[37]:


def recommend(Name):
    movies_id=Movies[Movies.Title==Name]['ids'].values[0]
    scores=list(enumerate(cosine_similarity[movies_id]))
    sorted_scores=sorted(scores,key=lambda x:x[1], reverse=True)
    sorted_scores=sorted_scores[1:]
    movies=[Movies[movies[0]==Movies['ids']]['Title'].values[0] for movies in sorted_scores]
    return movies


# In[38]:


def recommend_ten(movies_list):
    first_ten=[]
    count=0
    for movies in movies_list:
        if count > 9:
            break
        count += 1
        first_ten.append(movies)
    return first_ten


# In[39]:


test=recommend('Sing')
output=recommend_ten(test)
output


# __OUR SYSTEM RECOMMENDED 10 MOVIES BASED ON OUR TEST INPUT OF THE MOVIE "SING"__

# In[40]:


conda install -c conda-forge wordcloud


# In[41]:


##Wordcloud for Title
from wordcloud import WordCloud, STOPWORDS
comment_words = ''
stopwords = set(STOPWORDS)

for val in Movies.Title:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens)+" "
wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='white',
                      stopwords = stopwords,
                      min_font_size = 10).generate(comment_words)

# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# In[ ]:




