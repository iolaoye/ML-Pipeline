#!/usr/bin/env python
# coding: utf-8

# In[18]:


#This is a Natural language Processing project "in progress"
#This project is the Stage_02 of the ETL-ML pipeline for processing disaster data (text message) 
# for imediate and adequate response
#In Stage_01, I built an ETL pipeline for processing the data, cleaning the data, and saving it in sqlite database.  
#Now the clean data will be used in an ML pipeline for predicting the category of data for each text message.
#There are 36 different categories a text mesaage can be classified so it is a multioutput classification
#Staeg_03 will be hyperparameter tunning, and model testing. After deployment, any message entered by various individuals during a disaster
#will be classified for immediate and adequate response. 

# import some libraries
import sqlite3
import pandas as pd
import seaborn as sns
import re


# In[19]:


# Connect to the database and load data from database
   
conn = sqlite3.connect('clean_disasterC.db')
df = pd.read_sql('SELECT * FROM MCat', conn)
df = df.astype(str)
Xfeatures = df.iloc[:,1]
ylabels = df.iloc[:,4:]
X=Xfeatures
y=ylabels


# In[20]:


df


# In[21]:


Xfeatures


# In[22]:


ylabels


# In[23]:


#Tokenization function to process thetext data
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# Write the regular expression to detect the url in the text 
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


# In[24]:


#Tokenization function to process the text data
def tokenize(text):
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text =text.replace(url, "website")

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# test out function

for message in X[:]:
    tokens = tokenize(message)
    print(message)
    print(tokens, '\n')


# In[30]:


#Build a machine learning pipeline
#This machine pipeline takes in the message column as input and output classification results on the other 36 categories in the dataset. 
#using MultiOutputClassifier algorithm for predicting multiple target variables.

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,plot_confusion_matrix
#from sklearn.cross_validation import train_test_split
nltk.download(['punkt', 'wordnet'])
nltk.download('omw-1.4')

# import the Transformers
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(random_state=42, class_weight="balanced"))),
    
])

 


# In[26]:


#Train pipeline
#Split data into train and test sets
#Train pipeline
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=25)


# In[27]:


# Fit on Dataset
pipeline.fit(x_train, y_train)


# In[28]:



# predict on test data
print(x_test.iloc[0])
print("Actual Prediction:",y_test.iloc[0])


# In[29]:


# predict on test data
x_test=x_test.iloc[:]
y_pred=y_test.iloc[:]
x_test, y_pred

