#!/usr/bin/env python
# coding: utf-8

# ## Import Statements

# In[3]:


from requests_html import HTMLSession,PyQuery as pq
session=HTMLSession()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.downloader.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

import numpy as np
from datetime import datetime,timedelta
import json


# ## Testing SentimentIntensityAnalyzer

# In[ ]:


sia.polarity_scores('At least nine people were killed in the first strike when missiles destroyed a moving vehicle in the North Waziristan tribal region, the officials said.')


# ## Data Scraping 

# In[ ]:


start_date=datetime(2011,1,1)
end_date=datetime.now().date()
dt=timedelta(days=1)
f=open('/Users/maharshichattopadhyay/Documents/Study/Major_Project/JSON/headlines.json','w',encoding='utf-8')
all_data={}
for i in np.arange(start_date,end_date,dt).astype(datetime):  
    while 1:
        temp=f'https://www.dawn.com/archive/{str(i.date())}'
        r=session.get(temp)
        articles=r.html.find('article')
        if len(articles)>0:
            break
    print(len(articles))
    all_data[str(i.date())]=''
    for article in articles:
        t=pq(article.html)
        #heading_text=t('.theme--pakistan').siblings('h2').text()
        heading_text=t('.theme--india').siblings('h2').text()
        if heading_text:
            all_data[str(i.date())] += heading_text + '.'
json.dump(all_data,f,ensure_ascii=False) 
f.close()


# In[ ]:


r2=session.get('https://www.dawn.com/archive/2011-02-10')
a=r2.html.find('article')


# In[ ]:




