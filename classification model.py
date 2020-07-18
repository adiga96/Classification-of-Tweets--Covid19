# Document Matrix Creation for important Unigrams

countVectorizer = CountVectorizer(analyzer=lemmatizer) 
countVector = countVectorizer.fit_transform(shortlabel['Tweet_lemmatized'])
# countVector = pd.concat([tweets_text,countVector1])
print('{} Number of tweets has {} words'.format(countVector.shape[0], countVector.shape[1]))
#print(countVectorizer.get_feature_names())

count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())

# count_vect_df.head()
matrix_df = count_vect_df[['shortage','ppe','mask','hospital','toilet','ventilator','paper','food','gown']]
# countVector = pd.concat([tweets_text,countVector1])
matrix_df.loc['Column_Total']= matrix_df.sum(numeric_only=True, axis=0)
matrix_df


# Classification Models
# Random Tweets = 0, Medical Shortage = 1, Other Shortage = 2
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
labeled_ex

# Training Data and Test Data preparation
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

print("training set :",x_train.shape,y_train.shape)
print("testing set :",x_test.shape,y_test.shape)


print(x_train_counts.shape)
print(x_train_tfidf.shape)

print(x_test_counts.shape)
print(x_test_tfidf.shape)


# Support Vector Machine (SVM) ------------------------------

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
# model_SVM = OneVsRestClassifier(svm.SVC(gamma=0.01, C=1.0, probability=True, class_weight='balanced', kernel='linear', degree=2))
from sklearn.svm import LinearSVC

model_SVM = OneVsRestClassifier(LinearSVC())
# classifier.fit(vectorised_train_documents, train_labels)
# model_SVM = svm.SVC(kernel='linear')
model_SVM.fit(x_train_tfidf,y_train)
predictions_SVM = model_SVM.predict(x_test_tfidf)

# Create a confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

CM_SVM = confusion_matrix(y_test,predictions_SVM)
CM_SVM

# Classification Report
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions_SVM))
print()
print("Classification Report")
print(classification_report(y_test, predictions_SVM))


# ##### F1 Score
from sklearn.metrics import f1_score

SVM_F1 = f1_score(y_test,predictions_SVM,pos_label='positive',
                                           average='weighted')*100
SVM_F1.round(4)


# Accuracy
from sklearn.metrics import accuracy_score

SVM_accuracy = accuracy_score(y_test,predictions_SVM)*100
SVM_accuracy.round(4)


#Precision
from sklearn.metrics import precision_score
SVM_precision = precision_score(y_test,predictions_SVM,pos_label='positive',
                                           average='weighted')*100
SVM_precision.round(4)

# Recall
from sklearn.metrics import recall_score
SVM_recall = recall_score(y_test,predictions_SVM,pos_label='positive',
                                           average='weighted')*100
SVM_recall.round(4)

# Random Forest Classifier-------------------------------------

from sklearn.ensemble import RandomForestClassifier
model_RF = RandomForestClassifier(n_estimators=200)
model_RF.fit(x_train_tfidf,y_train)
predictions_RF = model_RF.predict(x_test_tfidf)

# To create a confusion matrix
CM_RF = confusion_matrix(y_test,predictions_RF)
CM_RF

# Classification Report
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions_RF))
print("Classification Report")
print(classification_report(y_test, predictions_RF))

RF_F1 = f1_score(y_test,predictions_RF,pos_label='positive',
                                           average='weighted')*100
RF_F1.round(4)

RF_accuracy = accuracy_score(y_test,predictions_RF)*100
RF_accuracy.round(4)

RF_precision = precision_score(y_test,predictions_RF,pos_label='positive',
                                           average='weighted')*100
RF_precision.round(4)

RF_recall = recall_score(y_test,predictions_RF,pos_label='positive',
                                           average='weighted')*100
RF_recall.round(4)


# Logistic Regression -------------------------------------------------------

from sklearn.linear_model import LogisticRegression
model_Log = LogisticRegression(random_state=400)
model_Log.fit(x_train_tfidf,y_train)
predictions_Log = model_Log.predict(x_test_tfidf)

from sklearn.metrics import confusion_matrix,f1_score
CM_Log = confusion_matrix(y_test,predictions_Log)


#Classification Report
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions_Log))
print("Classification Report")
print(classification_report(y_test, predictions_Log))


Log_F1 = f1_score(y_test,predictions_Log,pos_label='positive',
                                           average='weighted')*100
Log_F1.round(4)

Log_accuracy = accuracy_score(y_test,predictions_Log)*100
Log_accuracy.round(4)

Log_precision = precision_score(y_test,predictions_Log,pos_label='positive',
                                           average='weighted')*100
Log_precision.round(4)

Log_recall = recall_score(y_test,predictions_Log,pos_label='positive',
                                           average='weighted')*100
Log_recall.round(4)

#  Gradient Boosting Regressor ------------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
model_GB  = GradientBoostingClassifier(n_estimators=1000, learning_rate = 0.25, max_features=2, max_depth = 2, random_state = 0)
model_GB.fit(x_train_tfidf,y_train)
predictions_GB = model_GB.predict(x_test_tfidf)

# #### Classification Report
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions_GB))
print("Classification Report")
print(classification_report(y_test, predictions_GB))

GB_F1 = f1_score(y_test,predictions_GB,pos_label='positive',
                                           average='weighted')*100
GB_F1.round(4)

GB_accuracy = accuracy_score(y_test,predictions_GB)*100
GB_accuracy.round(4)


GB_precision = precision_score(y_test,predictions_GB,pos_label='positive',
                                           average='weighted')*100
GB_precision.round(4)

GB_recall = recall_score(y_test,predictions_GB,pos_label='positive',
                                           average='weighted')*100
GB_recall.round(4)

# heatmap of confusion matrix SVM/RF/Log/GB
cm_plt = pd.DataFrame(CM_SVM[:73])
plt.figure(figsize = (6,6))
ax = plt.axes()
sns.heatmap(cm_plt, annot=True)
ax.xaxis.set_ticks_position('top')
plt.show()
