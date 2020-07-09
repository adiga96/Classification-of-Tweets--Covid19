#!/usr/bin/env python
# coding: utf-8

# # Social media analysis of disruptiveness in the medical supply chain due to global pandemic #COVID 19

# ### IE 515: Transportation Analytics Project
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string
import re


# In[2]:


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

# In[5]:


def remove_punct(text):
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|(_[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    return text

tweets_text ['text_punc'] = tweets_text ['text'].apply(lambda x: remove_punct(x))
# tweets_text .head(10) <-------------------To Print results


# #### 2. Tokenization, split and covnert all to lowercase

# In[6]:


def tokenization(text):
    text = re.split('\W+', text)
    return text

tweets_text['Tweet_tokenized'] = tweets_text['text_punc'].apply(lambda x: tokenization(x.lower()))
# tweets_text.head() <-------------------To Print results


# #### 3. Removing Stop Words

# In[7]:


stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
tweets_text['text_nonstop'] = tweets_text['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))
# tweets_text.head(10) <-------------------To Print results


# #### 4. Stemming and Lemmitization
# 
# #### Please read for more details:
# 
# https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html

# #### 4.1 Stemming

# In[8]:


ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

tweets_text['Tweet_stemmed'] = tweets_text['text_nonstop'].apply(lambda x: stemming(x))

# tweets_text.head() <-------------------To Print results


# #### 4.2 Lemmatizing

# In[9]:


wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

tweets_text['Tweet_lemmatized'] = tweets_text['text_nonstop'].apply(lambda x: lemmatizer(x))

# tweets_text.head()  <------ To Print results


# ## Filtering Shortage related tweets 
# (from120K tweets)

# In[10]:


# tweets_text.fillna(0)
a = tweets_text[tweets_text['text'].str.contains('shortage', case=False)]
a.fillna(0)
shortlabel = a[["id","location","date","text","Tweet_lemmatized"]]
shortlabel = shortlabel.fillna(0)
# shortlabel


# ## To count most common words used in tweets
# 
# 1. More about Library (collections):
# https://docs.python.org/3/library/collections.html

# In[113]:


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


# In[114]:


my_list.pop(0)
x = list(map(lambda x: x[0], my_list))
y = list(map(lambda x: x[1], my_list))


plt.title("Important UNIGRAMS", size = 18)
plt.xlabel("UNIGRAMS", size = 12)
plt.ylabel("COUNT",size = 12)

plt.bar(x,y)
plt.rcParams['figure.figsize']= [12,10]


# ### Mask Shaped Word Cloud

# In[14]:


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


# ### Word Cloud

# In[15]:


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


# ## Spatial Analysis of Tweets

# In[20]:


#Initialisation of tweets with geolocations

mapp = pd.read_csv('df.csv',low_memory = False)
mapp = mapp.dropna()


# In[21]:


#cleaning text for Folium POP-UP
import re
def remove_punct(text):
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|(_[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    return text

mapp['cleaned_text'] = mapp['Text'].apply(lambda x: remove_punct(x))


# In[111]:


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


# ## Document Matrix Creation for important Unigrams

# In[115]:


countVectorizer = CountVectorizer(analyzer=lemmatizer) 
countVector = countVectorizer.fit_transform(shortlabel['Tweet_lemmatized'])
# countVector = pd.concat([tweets_text,countVector1])
print('{} Number of tweets has {} words'.format(countVector.shape[0], countVector.shape[1]))
#print(countVectorizer.get_feature_names())


# In[24]:


count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())

# count_vect_df.head()
matrix_df = count_vect_df[['shortage','ppe','mask','hospital','toilet','ventilator','paper','food','gown']]

# countVector = pd.concat([tweets_text,countVector1])
matrix_df.loc['Column_Total']= matrix_df.sum(numeric_only=True, axis=0)
matrix_df


# # Classification
# 
# Random Tweets = 0, Medical Shortage = 1, Other Shortage = 2

# In[67]:


# Load the Labeled data set 

def load_label_data():
    labeled_data = pd.read_csv('XX.csv',low_memory = False)
    return labeled_data

labeled_ex = load_label_data()

labeled_ex.label = labeled_ex.label.astype(int)
labeled_ex = labeled_ex.fillna(0)

targets = labeled_ex['label']
# Text_1 = labeled_ex['Text']
Text_1 = labeled_ex["Tweet_lemmatized"]


# In[68]:


labeled_ex


# ## Training Data and Test Data preparation

# In[69]:


# Preparation of TRAINING Data (80%) and TEST Data (20%)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Text_1, targets, test_size = 0.2, random_state = 42)


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)

x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)


# In[70]:


print("training set :",x_train.shape,y_train.shape)
print("testing set :",x_test.shape,y_test.shape)


# In[71]:


print(x_train_counts.shape)
print(x_train_tfidf.shape)


# In[72]:


print(x_test_counts.shape)
print(x_test_tfidf.shape)


# ##  Support Vector Machine (SVM) 

# #### SVM Model Fitting

# In[73]:


from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
# model_SVM = OneVsRestClassifier(svm.SVC(gamma=0.01, C=1.0, probability=True, class_weight='balanced', kernel='linear', degree=2))
from sklearn.svm import LinearSVC

model_SVM = OneVsRestClassifier(LinearSVC())
# classifier.fit(vectorised_train_documents, train_labels)
# model_SVM = svm.SVC(kernel='linear')
model_SVM.fit(x_train_tfidf,y_train)
predictions_SVM = model_SVM.predict(x_test_tfidf)


# In[122]:


# Create a confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

CM_SVM = confusion_matrix(y_test,predictions_SVM)
CM_SVM


# In[75]:


# # heatmap of confusion matrix SVM

# cm_plt = pd.DataFrame(CM_SVM[:73])
# plt.figure(figsize = (6,6))
# ax = plt.axes()
# sns.heatmap(cm_plt, annot=True)
# ax.xaxis.set_ticks_position('top')
# plt.show()


# #### Classification Report

# In[76]:


print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions_SVM))
print()
print("Classification Report")
print(classification_report(y_test, predictions_SVM))


# ##### F1 Score

# In[77]:


from sklearn.metrics import f1_score

SVM_F1 = f1_score(y_test,predictions_SVM,pos_label='positive',
                                           average='weighted')*100
SVM_F1.round(4)


# #### Accuracy

# In[78]:


from sklearn.metrics import accuracy_score

SVM_accuracy = accuracy_score(y_test,predictions_SVM)*100
SVM_accuracy.round(4)


# #### Precision

# In[79]:


from sklearn.metrics import precision_score

SVM_precision = precision_score(y_test,predictions_SVM,pos_label='positive',
                                           average='weighted')*100
SVM_precision.round(4)


# #### Recall

# In[80]:


from sklearn.metrics import recall_score

SVM_recall = recall_score(y_test,predictions_SVM,pos_label='positive',
                                           average='weighted')*100
SVM_recall.round(4)


# ## Random Forest Classifier
# 
# https://medium.com/analytics-vidhya/twitter-sentiment-analysis-8ef90df6579c
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# 
# #### RF Model Fitting
# 
# 
# 

# In[81]:


from sklearn.ensemble import RandomForestClassifier

model_RF = RandomForestClassifier(n_estimators=200)
model_RF.fit(x_train_tfidf,y_train)
predictions_RF = model_RF.predict(x_test_tfidf)


# In[82]:


# To create a confusion matrix
CM_RF = confusion_matrix(y_test,predictions_RF)
CM_RF


# In[83]:


# # heatmap of confusion matrix RF
# cm_plt = pd.DataFrame(CM_RF[:73])
# plt.figure(figsize = (5,5))
# ax = plt.axes()
# sns.heatmap(cm_plt, annot=True)
# ax.xaxis.set_ticks_position('top')
# plt.show()


# #### Classification Report

# In[84]:


print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions_RF))
print()
print("Classification Report")
print(classification_report(y_test, predictions_RF))


# #### F1 Score

# In[85]:


RF_F1 = f1_score(y_test,predictions_RF,pos_label='positive',
                                           average='weighted')*100
RF_F1.round(4)


# #### Accuracy

# In[86]:


RF_accuracy = accuracy_score(y_test,predictions_RF)*100
RF_accuracy.round(4)


# #### Precision

# In[87]:


RF_precision = precision_score(y_test,predictions_RF,pos_label='positive',
                                           average='weighted')*100
RF_precision.round(4)


# #### Recall

# In[88]:


RF_recall = recall_score(y_test,predictions_RF,pos_label='positive',
                                           average='weighted')*100
RF_recall.round(4)


# ## Logistic Regression

# In[89]:


from sklearn.linear_model import LogisticRegression

model_Log = LogisticRegression(random_state=400)
model_Log.fit(x_train_tfidf,y_train)


# In[90]:


predictions_Log = model_Log.predict(x_test_tfidf)


# In[91]:


from sklearn.metrics import confusion_matrix,f1_score
CM_Log = confusion_matrix(y_test,predictions_Log)


# In[92]:


# cm_plt = pd.DataFrame(CM_Log[:73])

# plt.figure(figsize = (5,5))
# ax = plt.axes()

# sns.heatmap(cm_plt, annot=True)
# ax.xaxis.set_ticks_position('top')
# plt.show()


# #### Classification Report

# In[93]:


print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions_Log))
print()
print("Classification Report")
print(classification_report(y_test, predictions_Log))


# In[94]:


Log_F1 = f1_score(y_test,predictions_Log,pos_label='positive',
                                           average='weighted')*100
Log_F1.round(4)


# In[95]:


Log_accuracy = accuracy_score(y_test,predictions_Log)*100
Log_accuracy.round(4)


# In[96]:


Log_precision = precision_score(y_test,predictions_Log,pos_label='positive',
                                           average='weighted')*100
Log_precision.round(4)


# In[97]:


Log_recall = recall_score(y_test,predictions_Log,pos_label='positive',
                                           average='weighted')*100
Log_recall.round(4)


# ## Gradient Boosting Regressor

# In[98]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
model_GB  = GradientBoostingClassifier(n_estimators=1000, learning_rate = 0.25, max_features=2, max_depth = 2, random_state = 0)
model_GB.fit(x_train_tfidf,y_train)


# In[99]:


predictions_GB = model_GB.predict(x_test_tfidf)


# In[100]:


# CM_GB = confusion_matrix(y_test,predictions_GB)
# cm_plt = pd.DataFrame(CM_GB[:100])
# plt.figure(figsize = (6,6))
# ax = plt.axes()
# sns.heatmap(cm_plt, annot=True)
# ax.xaxis.set_ticks_position('top')
# plt.show()


# #### Classification Report

# In[101]:


print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions_GB))
print()
print("Classification Report")
print(classification_report(y_test, predictions_GB))


# In[102]:


GB_F1 = f1_score(y_test,predictions_GB,pos_label='positive',
                                           average='weighted')*100
GB_F1.round(4)


# In[103]:


GB_accuracy = accuracy_score(y_test,predictions_GB)*100
GB_accuracy.round(4)


# In[104]:


GB_precision = precision_score(y_test,predictions_GB,pos_label='positive',
                                           average='weighted')*100
GB_precision.round(4)


# In[105]:


GB_recall = recall_score(y_test,predictions_GB,pos_label='positive',
                                           average='weighted')*100
GB_recall.round(4)





# ### Comparison Chart of Scores

# In[66]:


# set width of bar
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






