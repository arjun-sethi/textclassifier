from operator import itemgetter
import nltk
from nltk.corpus import brown
from nltk.tag import UnigramTagger

import numpy as np
import glob
import os

import string
import math

#for the built in sklearn
from sklearn.feature_extraction.text import TfidfTransformer


tests={}
trains={}
bag_of_words = {}
inverse_frequency ={}
tfidf = {}
categories=["adventure","editorial","fiction","government","hobbies","humor","learned","mystery","news","religion","reviews","romance","science_fiction"]
for category in categories:
	fileids=brown.fileids(category)
	num_of_trains=math.ceil(0.6*len(fileids))
	fileids_train=[]
	fileids_test=[]
	for x in range (0,num_of_trains):
		fileids_train.append(fileids[x])
	for x in range (num_of_trains,len(fileids)):
		fileids_test.append(fileids[x])
	trains[category]=fileids_train
	tests[category]=fileids_test

##BAG OF WORDS
# for every category in brown corpus, count frequency of distinct words in each category
for category in categories:
	bag_of_words[category]={}
	for word in brown.words(fileids=trains[category]):
		if word.lower() in bag_of_words[category].keys():
			bag_of_words[category][word.lower()]+=1
		else:
			bag_of_words[category][word.lower()]=1

#remove punctuation
for category in bag_of_words.keys():
	for punct in string.punctuation:
		if punct in bag_of_words[category]: del bag_of_words[category][punct]
	for punct in ["''","``","--"]:
		if punct in bag_of_words[category]: del bag_of_words[category][punct]

#at this point, termfrequency=bag_of_words[category][word]
#get inverse-document frequency. log(# of documents / # documents containing the term)
for category in bag_of_words.keys():
	for word in bag_of_words[category].keys():	#for every word in every category
		if word not in inverse_frequency.keys():
			inverse_frequency[word]=1
		else:
			inverse_frequency[word]+=1

for word in inverse_frequency.keys():
	inverse_frequency[word]=math.log((len(bag_of_words.keys())/inverse_frequency[word])+1)

#calculate termfreq*inversefreq
for category in bag_of_words.keys():
	for word in bag_of_words[category].keys():
		#mutliply term frequency by inverse term frequency
		bag_of_words[category][word]=bag_of_words[category][word]*inverse_frequency[word]

#write word : frequency to file, named by category
for category in bag_of_words.keys():
	file=open(category+".txt","w")
	for word in bag_of_words[category].keys():
		file.write(word+" :\t"+str(bag_of_words[category][word])+"\n")
	file.close()

##POS TAGGING COUNTS
testFile=[]

testFile=brown.words(fileids=tests["news"])

probability ={}

for category in bag_of_words.keys():
	probability[category]=1
	for word in testFile:
		if word in bag_of_words[category].keys():
			probability[category]*=bag_of_words[category][word]

for category in probability:
	probability[category]=(1/15)*probability[category]*100
	#print(category+" : "+str(probability[category]))

################################################
allwords=[]
for category in bag_of_words.keys():
	for word in bag_of_words[category].keys():
		if word not in allwords:
			allwords.append(word)

freq_term_matrix=[]
for category in bag_of_words.keys():
	new=[]
	for column in range(0,len(allwords)):
		if allwords[column] in bag_of_words[category].keys():
			new.append(bag_of_words[category][allwords[column]])
		else:
			new.append(0)
	freq_term_matrix.append(new)

file=open("example.txt","w")
array=["can","could","may","might","must","will"]
file.write("\t")
for x in range(0,len(array)):
	file.write("\t"+array[x])

file.write("\n")
x=0
for category in bag_of_words.keys():
	file.write(category+"\t")
	for word in array:
		file.write(str(freq_term_matrix[x][allwords.index(word)])+"\t")
	file.write("\n")
	x+=1
file.close()


data=[]
targets=[]
i=0
for category in trains:
	for x in range(0,len(trains[category])-1):
		mystring=' '.join(brown.words(fileids=[trains[category][x]]))
		data.append(mystring.translate(str.maketrans('','',string.punctuation)))
		targets.append(category)

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer



from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
text_clf = text_clf.fit(data, targets)

dataTest=[]
targetsTest=[]
for category in tests:
	for x in range(0,len(tests[category])-1):
		mystring=' '.join(brown.words(fileids=[tests[category][x]]))
		mystring=mystring.translate(str.maketrans('','',string.punctuation))
		dataTest.append(mystring)
		targetsTest.append(category)
predicted = text_clf.predict(dataTest)
print(len(dataTest))
np.mean(predicted == targetsTest)
print(np.mean(predicted == targetsTest)*100)
