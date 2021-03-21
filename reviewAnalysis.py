
#Importing Libraries 
import nltk
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet

#nltk.download('sentiwordnet')

#Importing Data 
data1 = open('amazon_cells_labelled.txt') 
text = data1.read() # Reading text

sent = re.findall(r'.*\n',text) 
sent[:10] # seperating each review with label and output only first 10 results.


sent = [(line[:-3],int(line[-2])) for line in sent]
sent[:10] # seperating each review and its label and storing in tuple

#Converting data from text to Data Frame

data = pd.DataFrame(sent,columns=['review','score'])
print(data)

y = data['score']

print(sum(data['score']))

#Pre-processing
def pre_process(text):
    """Removing punctuations, numbers and stopwords / Unwanted tokens"""
    clean_text = [char for char in text if char not in string.punctuation] # removing punctuation
    clean_text = "".join(clean_text)
    clean_text = clean_text.lower()  # converting to lower-case
    clean_text = re.sub("\d+", "", clean_text) # removing numbers
    clean_text = [words for words in clean_text.split(" ") if words not in stopwords.words('english')] # removing english stopwords
    return clean_text


x = CountVectorizer(analyzer=pre_process).fit(data['review'])
x  # converting text document to vector tokens count


x = x.transform(data['review'])
print(x) # vector representation of reviews

#Data Splitting

# x = vector representation of reviews
# y = vector representation of score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1,
            random_state = 700) #training data 90% and testing data 10%


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# Using Naive_bayes

nb = MultinomialNB()
nb.fit(x_train,y_train)

#Accuracy 
pred = nb.predict(x_test)
print("ACCURACY : ", accuracy_score(y_test,pred))


# Method for testing
def test():
    
    """
    Using wordnet and sentiwordnet, we are calculating positive and negative score of
    input string / review. And showing total score (positive or negative) on the basis
    of prediction.
    
    """
    
    test = input("Enter a sentence to be checked veview:  ")
    tokens = test.split(' ')
    pos_total = 0
    neg_total = 0
    for t in tokens:
        syn_t = wordnet.synsets(t)
        if len(syn_t) > 0:
            syn_t = syn_t[0]
            senti_syn_t = sentiwordnet.senti_synset(syn_t.name())
            if senti_syn_t.pos_score() > senti_syn_t.neg_score():
                pos_total += senti_syn_t.pos_score()
            else:
                neg_total += senti_syn_t.neg_score()
    total_score = pos_total - neg_total

    x = CountVectorizer(analyzer = pre_process).fit(data['review'])
    test=x.transform([test])

    if (nb.predict(test)[0] == 1):
        return("\nTrue review with score of ", total_score)
    else:
        return("\nFake review with score of ", total_score)

# Test case #1
### Input string: "It is the best charger I have seen on the market yet."


comment, score = test()
print(comment, score)

# Test case #2
### Input string: "you could only take 2 videos at a time and the quality was very poor."

comment, score = test()
print(comment, score)

# Using Logistic Regression  
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred_train = logreg.predict(x_train)
print("Training Accuracy: ", logreg.score(x_train, y_train))


y_pred_test = logreg.predict(x_test)
print("Test Accuracy: ", logreg.score(x_test, y_test))

# Test case #1
### Input string: "It fits so securely that the ear hook does not even need to be used and the sound is better directed through your ear canal."

comment, score = test()
print(comment, score)
