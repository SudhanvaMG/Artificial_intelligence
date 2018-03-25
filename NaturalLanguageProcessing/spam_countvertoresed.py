import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

ds = pd.read_csv("spam.csv")
ds.loc[ds["v1"]=="ham","v1"]=0
ds.loc[ds["v1"]=="spam","v1"]=1

x = ds["v2"]
y = ds["v1"]

x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
x_train = [unicode(strd, errors='replace') for strd in x_train ]
x_test = [unicode(strd, errors='replace') for strd in x_test ]
#cv = CountVectorizer()
#how the countvectorizer works
#tes = cv.fit_transform(["hello , welcome to my hello program","welcome to my hello worlds"])
#tes.toarray()
#cv.get_feature_names()
cv1 = TfidfVectorizer(min_df=1,stop_words="english")
x_traincv = cv1.fit_transform(x_train)
x_testcv = cv1.transform(x_test)

mnb = MultinomialNB()
y_train = y_train.astype('int')

mnb.fit(x_traincv, y_train)
pred = mnb.predict(x_testcv)

actual = np.array(y_test)
count = 0
for i in range(len(pred)):
    if pred[i]==actual[i]:
        count = count+1
        
acc = len(pred)/count