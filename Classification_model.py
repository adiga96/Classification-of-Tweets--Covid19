#Social media analysis of disruptiveness in the medical supply chain due to global pandemic #COVID 19

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string
import re

#loading main dataset (150K tweets)
def load_data():
    data = pd.read_csv('Final.csv',low_memory = False)
    return data

# In[3]:
data = load_data()
data = data.drop_duplicates('id')

# Total tweets left ~ 120K
tweets = data
# In[4]:
tweets_text  = pd.DataFrame(tweets[['id','location','date','text']])


# ## Pre processing text Data

# #### 1. Removing punctuations, @user, https:// and unwanted characters from the tweets

def remove_punct(text):
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|(_[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    return text
    
# 2. Tokenization, split and covnert all to lowercase    
def tokenization(text):
    text = re.split('\W+', text)
    return text  

# 3. Removing Stop Words    
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
 
# 4. Stemming and Lemmitization    
ps = nltk.PorterStemmer()
def stemming(text):
    text = [ps.stem(word) for word in text]
    return text    

wn = nltk.WordNetLemmatizer()
def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text    
    
tweets_text ['text_punc'] = tweets_text ['text'].apply(lambda x: remove_punct(x))
tweets_text['Tweet_tokenized'] = tweets_text['text_punc'].apply(lambda x: tokenization(x.lower()))
tweets_text['text_nonstop'] = tweets_text['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))
tweets_text['Tweet_stemmed'] = tweets_text['text_nonstop'].apply(lambda x: stemming(x))
tweets_text['Tweet_lemmatized'] = tweets_text['text_nonstop'].apply(lambda x: lemmatizer(x))


# ## Filtering Shortage related tweets 
# (from120K tweets)
# tweets_text.fillna(0)
a = tweets_text[tweets_text['text'].str.contains('shortage', case=False)]
a.fillna(0)
shortlabel = a[["id","location","date","text","Tweet_lemmatized"]]
shortlabel = shortlabel.fillna(0)


import collections
def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt
#  .most_common( 'enter your desired number')
my_list = shortlabel[['Tweet_lemmatized']].apply(count_words)['Tweet_lemmatized'].most_common(9) 
my_list

my_list.pop(0)
x = list(map(lambda x: x[0], my_list))
y = list(map(lambda x: x[1], my_list))

plt.title("Important UNIGRAMS", size = 18)
plt.xlabel("UNIGRAMS", size = 12)
plt.ylabel("COUNT",size = 12)
plt.bar(x,y)
plt.rcParams['figure.figsize']= [12,10]


# Mask Shaped Word Cloud
from wordcloud import WordCloud,STOPWORDS
 
from PIL import Image
import numpy as np
my_list_WC = shortlabel[['Tweet_lemmatized']].apply(count_words)['Tweet_lemmatized'].most_common(1000) 
unique_string= ','.join(str(v) for v in my_list_WC)
twitter_mask= np.array(Image.open('mask.png')) #sitr.jpg image name
wCloud= WordCloud(width=1800, height=1400, background_color='white', 
                  mask=twitter_mask,stopwords=STOPWORDS).generate(unique_string)
plt.imshow(wCloud)
plt.axis("off")
plt.savefig("WordCloud1.png",dpi=300) #save img with name that u want

#Word Cloud
my_list_WC = shortlabel[['Tweet_lemmatized']].apply(count_words)['Tweet_lemmatized'].most_common(1000) 

from wordcloud import WordCloud 
#convert list to string and generate
unique_string= ','.join(str(v) for v in my_list_WC)
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
plt.figure(figsize=(15,8))

plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("WordCloud2.png", bbox_inches='tight')
plt.show()
plt.close()


# Spatial Analysis of Tweets
#Initialisation of tweets with geolocations
mapp = pd.read_csv('df.csv',low_memory = False)
mapp = mapp.dropna()

# In[21]:
#cleaning text for Folium POP-UP
mapp['cleaned_text'] = mapp['Text'].apply(lambda x: remove_punct(x))

import folium

# Initalisation of Folium Map
mapit = folium.Map( location=[52.667989, -1.464582], zoom_start=3)  #set Zoom Number to any value based on amount of zoom required
# Bounding Box for United States
folium.vector_layers.Rectangle([[49.95121990866204, -130.51757812500003], [24.126701958681682, -130.25390625000003],                                [24.766784522874453, -54.93164062500001], [50.401515322782366, -55.54687500000001]],
                               color = '#228B22',popup = 'United States', tooltip=None).add_to(mapit)
                               
#---------------------------Bounding Box for top tweeted Cities in United States--------------

 #Bounding Box of New YorK - New Jersey - Washington                              
folium.vector_layers.Rectangle([[41.236511201246216, -74.27856445312501], [40.68896903762434, -73.25683593750001],                                 [38.976492485539424, -74.91577148437501], [39.50404070558415, -75.92651367187501]],
                               color = 'blue',popup=None, tooltip=None).add_to(mapit)
#Bounding Box for Los Angles Area
folium.vector_layers.Rectangle ([[34.452218472826566, -119.57519531250001], [34.43409789359469, -117.82836914062501],                                 [33.36723746583834, -117.81738281250001], [33.37641235124679, -119.55322265625001]],
                               color = 'blue',popup=None, tooltip=None).add_to(mapit)
#Bounding Box for Seattle Are
folium.vector_layers.Rectangle ([[47.06263847995432, -123.0908203125], [47.06263847995432, -121.90429687500001],                                 [48.66194284607008, -121.93725585937501], [48.66194284607008, -123.0908203125]],
                               color = 'blue',popup=None, tooltip=None).add_to(mapit)
# Loop over all the Tweets and Respective coordinates
for i in range(len(mapp)):
    lat = mapp['lat'][i]
    long = mapp['long'][i]
    pop = mapp['cleaned_text'][i]
    
    # To mark all tweets as circle markers
#     print(lat, long)
    folium.CircleMarker(location=[lat, long], popup=pop, fill_color='red' ,weight=1,color="black" ,radius=6 ).add_to(mapit)
#saving map in a html file
mapFile = 'osm.html'
mapit.save(mapFile)

#to display map
mapit

#Comparison Chart of Scores

barWidth = 0.15
 
# set height of bar
F1_bar = [SVM_F1, RF_F1, Log_F1, GB_F1]
Accuracy_bar = [SVM_accuracy, RF_accuracy, Log_accuracy, GB_accuracy]
Precision_Bar = [SVM_precision, RF_precision ,Log_precision ,GB_precision ]
Recall_bar =[SVM_recall, RF_recall , Log_recall, GB_recall]
 
# Set position of bar on X axis
r1 = np.arange(len(F1_bar))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
 
# Make the plot
plt.bar(r1, F1_bar, color='#0e2433', width=barWidth, edgecolor='white', label='SVM')
plt.bar(r2, Accuracy_bar, color='#1c4966', width=barWidth, edgecolor='white', label='Random Forest')
plt.bar(r3, Precision_Bar, color='#296d98', width=barWidth, edgecolor='white', label='Logistic Reg')
plt.bar(r4, Recall_bar, color='#3792cb', width=barWidth, edgecolor='white', label='Gradient Booster')
 
# Add xticks on the middle of the group bars
plt.xlabel('Model Scores',size= 15, fontweight='bold')
plt.xticks([r + barWidth for r in range(len(F1_bar))], ['F1 Score', 'Accuracy', 'Precision', 'Recall'])
 
# Create legend & Show graphic
plt.legend()
plt.legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)

plt.rcParams['figure.figsize']= [12,6]










