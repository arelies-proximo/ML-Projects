#%matplotlib inline
    #inline -> to show the graph on the page (On Jupyter) 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


from sklearn.datasets import fetch_20newsgroups
    
news_categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
  'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
   'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']



train = fetch_20newsgroups(subset='train', categories=news_categories)

test = fetch_20newsgroups(subset='test', categories=news_categories)



#importing necessary packages for tokenizing and understanding words
from sklearn.feature_extraction.text import TfidfVectorizer
    #for weighing the words, based on how many time used in doc
    #it's formula (most used)
    #word like in, a, the, or have less weights
    #words: criminal, millions, control have high weights

from sklearn.naive_bayes import MultinomialNB
    #Multinomial Naive Bayes

from sklearn.pipeline import make_pipeline
    #for getting the info from Vectorizer, pump it to MultiNB
    #pipeline-> for controlling the flow, how to move

#creating a model based on MultinomialNB
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

#training the model with the train datasets
model.fit(train.data, train.target)
    #train data goes into TfidfVectorizer
    #it weights all the words in there
    #1000 of words with weights
    #connects the combination of words and weights with 
    #respective targets

#creating labels for test data
labels = model.predict(test.data)
    #to predict the outcome now
    #how good the model is !
    #do the prediction (labels) match with real test.targets values


#Creating Confusion matrix and heat map
#Confusion matrix -> How Confused is our answer (Error percentage)
    #put that on a heat map (colors)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
xticklabels=train.target_names, yticklabels=train.target_names)
    #annot = Annotations (numbers 166,1,0 -> nos. in each cell)
    #fmt = Format

#plotting heatmap of confusion_matrix
plt.xlabel("True Label")
plt.ylabel("Predicted Label (By Model)")


#creating a function
#to categories new lines, articles 
#To find categories of new sentences, articles

def predict_category(sentence,  train=train, model=model):
        #train = training model
        #model -> pipeline

    predict = model.predict([sentence])
        #push string into pipeline, tokenze it, assign weights
        #then put it through naive bayes
        #predict the sentence

    print(train.target_names[predict[0]])
        #print the predicted target name (passing index)

