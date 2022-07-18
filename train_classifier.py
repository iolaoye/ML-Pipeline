#!/usr/bin/env python
# coding: utf-8

# In[20]:


#This is a Natural language Processing project "in progress"
#This project is the Stage_02 of the ETL-ML pipeline for processing disaster data (text message) 
# for imediate and adequate response
#In Stage _01, I built an ETL pipeline for processing the data, cleaning the data, and saving it in sqlite database.  
#Now the clean data will be used in an ML pipeline for predicting the category of data for each text message.
#There are  36 different possible categories a text mesaage can be classified so it is a multioutput classification
#After deployment, any message entered by various individuals during a disaster
#will be classified for immediate and adequate response. 

# import some libraries
import sqlite3
import pandas as pd
import seaborn as sns
import re


# In[21]:


# Connect to the database and load data from database
   
conn = sqlite3.connect('clean_disasterC.db')
df = pd.read_sql('SELECT * FROM MCat', conn)
df = df.astype(str)
Xfeatures = df.iloc[:,1]
ylabels = df.iloc[:,4:]
X=Xfeatures
y=ylabels


# In[22]:


df


# In[23]:


Xfeatures


# In[24]:


ylabels


# In[25]:


#Write a tokenization function to process your text data
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# Write the regular expression to detect the url in the text 
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


# In[26]:


#Write a Tokenization function to process the text data
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


# In[27]:


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

 


# In[28]:


#Train pipeline
#Split data into train and test sets
#Train pipeline
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=25)


# In[29]:


# Fit on Dataset
pipeline.fit(x_train, y_train)


# In[30]:


classifier =  MultiOutputClassifier(RandomForestClassifier(random_state=42, class_weight="balanced"))


# In[31]:



# predict on test data
print(x_test.iloc[0])
print("Actual Prediction:",y_test.iloc[0])


# In[32]:


# predict on test data
x_test=x_test.iloc[:]
y_pred=y_test.iloc[:]
x_test, y_pred


# In[33]:


scores = cross_validate(classifier, x_train, y_train, cv=2, scoring=['f1_weighted'])


# In[34]:


#Improve your model¶
#Use grid search to find better parameters.
parameters = {'learning_rate': [0.01,0.02,0.03],
               'subsample'    : [0.9, 0.5, 0.2],
               'n_estimators' : [100,500,1000],
               'max_depth'    : [4,6,8]
                 }


# In[35]:


from sklearn.model_selection import GridSearchCV
grid_pipeline = GridSearchCV(estimator=pipeline, param_grid = parameters, cv = 2, n_jobs=-1)
grid_pipeline.fit(x_train, y_train)


# In[ ]:


print(" Results from Grid Search " )
print("\n The best score across ALL searched params:\n",grid_pipeline.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_pipeline.best_params_)


# In[37]:


#Export the model as a pickle file
import pickle


# In[38]:


pickle.dump(pipeline, open('pipeline.pkl', 'wb'))


# In[40]:


pickled_pipeline = pickle.load(open('pipeline.pkl', 'rb'))
#pickled_pipeline.predict(x_test)

