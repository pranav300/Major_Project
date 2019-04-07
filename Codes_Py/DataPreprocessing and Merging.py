#!/usr/bin/env python
# coding: utf-8

# ### Merging the stock data and news info

# In[2]:


import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.downloader.download('vader_lexicon')
sia=SentimentIntensityAnalyzer()


# In[3]:


df_news = pd.read_json('/Users/maharshichattopadhyay/Documents/Study/Major_Project/JSON/headlines_converted.json',convert_dates=['Date'])
df_news = pd.DataFrame.from_records(df_news['news'])
df_news.head()


# In[6]:


df_news['Date'] = pd.to_datetime(df_news['Date'])
df_news['Date'] = df_news['Date'].dt.strftime('%d-%b-%Y')

df_bank = pd.read_csv('/Users/maharshichattopadhyay/Documents/Study/Major_Project/DataSet/Final_Dataset/MCB_01012010_06122018.csv')
df_bank.head()


# In[44]:


df_news.head()


# In[35]:


df_merged = df_bank.merge(df_news,on='Date')
df_merged.head()


# In[36]:


df_merged.tail()


# In[37]:


df_merged.shape


# In[38]:


df_merged=df_merged[['Date','Open','High','Low','Close','Volume','headlines']]
df_merged.head()


# In[39]:


df_merged['compund']=''
df_merged['neg']=''
df_merged['neu']=''
df_merged['pos']=''

df_merged.head()


# In[40]:


for index,sentence in enumerate(df_merged['headlines']):
    ps=sia.polarity_scores(sentence)
    df_merged['compund'][index]=ps['compound']
    df_merged['neg'][index]=ps['neg']
    df_merged['neu'][index]=ps['neu']
    df_merged['pos'][index]=ps['pos']
#   break


# In[41]:


final_df = df_merged[['Date','compund','neg','neu','pos','Open','High','Low','Close','Volume']]


# In[42]:


final_df.head()


# In[43]:


final_df.to_csv('final_data_mcb.csv',index=False)


# In[ ]:




