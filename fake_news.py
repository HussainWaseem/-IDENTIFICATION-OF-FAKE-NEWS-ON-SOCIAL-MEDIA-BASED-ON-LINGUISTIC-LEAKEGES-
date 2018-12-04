import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.get_backend()
matplotlib.matplotlib_fname()
matplotlib.style.use('ggplot')
from subprocess import check_output

fake_news = pd.read_csv("/Users/akash/Documents/MBA_BA-TERM4/Research_methods/Project_Fake_News/fake_or_real_news.csv",index_col="Unnamed: 0")
fake_news = fake_news.reset_index(drop=True)
print(fake_news.head())
print(fake_news.groupby(['label'])['text'].count())
fake_news['label'] = fake_news['label'].map({'FAKE': 0, 'REAL': 1})
print(fake_news.head())



from matplotlib import pyplot as plt
x=[-2,2]
y=[3164, 3171]

plt.bar(x,y)
#plt.xlabel('type of sentiment')
plt.ylabel('count')
plt.title('number of fake and true news')
plt.legend()
plt.show()



df = pd.DataFrame()
print (df.head())
df["text"] = fake_news['title'].map(str)+ " " + fake_news['text']
print (df.head())
print(df["text"])
df['label'] = fake_news['label']
print (df['label'])

df['text'] = df['text'].str.lower()
only_text = pd.DataFrame()
only_text = df['text']
print(only_text)

#removing stopwords
import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english') 
print (stop)
print (len(stop)) # corpus of 179 words
only_text = only_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print (only_text)

#data cleaning
#1. it is seen that there are hyperlinks present, so lets remove the hyperlinks
#also taking only those words which have length less than 15letters and greater than 2 into 'a'
#p=re.compile(r'\<http.+?\>', re.DOTALL)
a = []
b=[]
new_string = []
for i in only_text:
    #b =re.sub(p, '', i)    
    b = ' '.join([w for w in i.split() if len(w)<15])
    b = ' '.join([w for w in b.split() if len(w)>3])
    a.append(b)

#creating pandas dataframe and then converting it into panda series 
#removing all the characters except a-z cause we do not need numbers for sentiment analysis
#removing words am, pm
only_text_pd = pd.DataFrame({'text':a})
print (only_text_pd)
only_text_se = only_text_pd['text']
print (only_text_se)
only_text_se= only_text_se.str.replace('[^a-z \n]',"")
print (only_text_se)
only_text_se = only_text_se.str.replace('fake|real', "")
print (only_text_se)
only_text_se[23] #processed text
print (only_text_se)


#creating series of pandas
X= only_text_se
y= df['label']

#splitting the data into train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



#initiating count vectorizer and removing stop words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vect = CountVectorizer(stop_words='english',max_df=.3)
print (vect)
#fitting train data and then transforming it to count matrix
X_train_dtm = vect.fit_transform(X_train)
print (X_train_dtm)
X_train_dtm

#transforming the test data into the count matrix initiated for train data
#no fitting takes place
X_test_dtm = vect.transform(X_test)
X_test_dtm
print(X_test_dtm)


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

#fitting the model into train data 
nb.fit(X_train_dtm, y_train)

#predicting the model on train and test data
y_pred_class_test = nb.predict(X_test_dtm)
y_pred_class_train = nb.predict(X_train_dtm)

nb.predict(X_test_dtm)

X_test.iloc[1]


# calculate accuracy of class predictions
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class_test))
print(metrics.accuracy_score(y_train, y_pred_class_train))



# print the confusion matrix

from sklearn.metrics import confusion_matrix
cm = metrics.confusion_matrix(y_test, y_pred_class_test)
#plot_confusion_matrix(cm,classes=['FAKE', 'REAL'])

print (cm)

import seaborn as sns
import matplotlib
#matplotlib inline
cm = metrics.confusion_matrix(y_test, y_pred_class_test)
sns.set(font_scale=1.4)#for label size
sns.heatmap(cm,annot=True,annot_kws={"size": 16})# font size


y_test.value_counts()


# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)


# examine the first 50 tokens
print(X_train_tokens[0:50])


# examine the last 50 tokens
print(X_train_tokens[-50:])

# Naive Bayes counts the number of times each token appears in each class
print (nb.feature_count_)

# rows represent classes, columns represent tokens
print (nb.feature_count_.shape)

# number of times each token appears across all fake articles
fake_token_count = nb.feature_count_[0, :]
print (fake_token_count)


# number of times each token appears across all real articles
real_token_count = nb.feature_count_[1, :]
print (real_token_count)


# create a DataFrame of tokens with their separate real and fake counts
tokens = pd.DataFrame({'token':X_train_tokens, 'real':real_token_count, 'fake':fake_token_count}).set_index('token')

freq_real = tokens.sort_values(by='real', ascending = False)
freq_real = freq_real.drop('fake',1)
print (freq_real.head())

'''
from matplotlib import pyplot
import matplotlib.pyplot as ply
a4_dims = (13, 8.27)  #oo is the frequency of the words present in real records
fig, ax = pyplot.subplots(figsize=a4_dims)
g = sns.barplot(x="token",y='yes',data=oo,ax=ax)
g.set_xticklabels(rotation=30,labels=oo.token)
plt.title('Top 20 words present in opened records')
ax.set(xlabel='Words', ylabel='Frequency of word')
'''


freq_fake = tokens.sort_values(by='fake', ascending = False)
freq_fake = freq_fake.drop('real',1)
freq_fake = freq_fake.drop(freq_fake.index[[3]])
freq_fake.head()


new_text = pd.Series('during election time the world was expecting clinton party taking up the government but after the election results were out it is found that trump got the presidency and it is claimed that trump was aided by russian and media higher authorities to change the opinion of people')
new_test_dtm = vect.transform(new_text)
print (nb.predict(new_test_dtm))

print (nb.class_count_)

#Before we can calculate the "spamminess" of each token, we need to avoid dividing by zero and account for the class imbalance
# add 1 to real and fake counts to avoid dividing by 0
tokens['real'] = tokens.real + 1
tokens['fake'] = tokens.fake + 1
print (tokens.sample(5, random_state=6))

# convert the real  and fake counts into frequencies
tokens['real'] = tokens.real / nb.class_count_[0]
tokens['fake'] = tokens.fake / nb.class_count_[1]
tokens.sample(5, random_state=6)

# calculate the ratio of fake to real for each token
tokens['fake_ratio'] = tokens.fake / tokens.real
tokens.sample(5, random_state=6)

# calculate the ratio of fake to real for each token
tokens['real_ratio'] = tokens.real / tokens.fake
tokens.sample(5, random_state=6)

# examine the DataFrame sorted by fake_ratio
tokens_real = tokens.sort_values('real_ratio', ascending=False).head(5)
print (tokens_real)



# Logistic Regression

from sklearn.linear_model import LogisticRegression
seed=12
logreg = LogisticRegression(C=2, random_state=12, class_weight='balanced')
print (X_train_dtm)
print (y_train)
logreg.fit(X_train_dtm, y_train)


y_log_pred_test = logreg.predict(X_test_dtm)
y_log_pred_train = logreg.predict(X_train_dtm)

print(metrics.accuracy_score(y_test, y_log_pred_test))

print (metrics.accuracy_score(y_train, y_log_pred_train))
print (metrics.confusion_matrix(y_test, y_log_pred_test))


# Random Forest

from sklearn.ensemble import RandomForestClassifier
RF_clf = RandomForestClassifier(n_estimators=25)
RF_clf.fit(X_train_dtm, y_train)


#predicting on train and test data
RF_clf.fit(X_train_dtm, y_train)
y_rf_pred_test = RF_clf.predict(X_test_dtm)
y_rf_pred_train = RF_clf.predict(X_train_dtm)
probs_rf = RF_clf.predict_proba(X_test_dtm)


probs_rf = RF_clf.predict_proba(X_test_dtm) #predict probabilities
print (probs_rf)


probs = probs_rf[:,1]  #taking a column

probs[probs > 0.5] =1  #setting thresholds
probs[probs <= 0.5] = 0
print (probs)
print (metrics.accuracy_score(y_test, probs))  #accuracy on the predicted probabilities


#good accuracy on test data
metrics.accuracy_score(y_test, y_rf_pred_test)  #accuracy on predicted test data


#overfitting on train dataa
metrics.accuracy_score(y_train, y_rf_pred_train)



