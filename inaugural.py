import nltk
from nltk.corpus import movie_reviews
import math
import string
from nltk.stem.snowball import SnowballStemmer
import random


tests={}
trains={}
bag_of_words = {}
train_documents=[]
train_categories=[]
test_documents=[]
test_categories=[]
stemmer = SnowballStemmer("english")
stop_words=[]
stop_words_nonstem=["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount", "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as", "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
for word in stop_words_nonstem:
	stop_words.append(stemmer.stem(word))
categories=movie_reviews.categories()
#categories=["humor","mystery","news","religion","reviews","romance","science_fiction"]

for category in categories:
	fileids=movie_reviews.fileids(category)
	num_of_trains=math.ceil(0.67*len(fileids))
	fileids_train=[]
	fileids_test=[]
	fileids_train=random.sample(fileids,num_of_trains)
	for x in range (0,num_of_trains):
		#fileids_train.append(fileids[x])
		#train_documents.append(fileids[x])
		train_documents.append(fileids_train[x])
		train_categories.append(category)
	#for x in range (num_of_trains,len(fileids)):
	for x in range (0,len(fileids)):
		if fileids[x] not in fileids_train:
			fileids_test.append(fileids[x])
			test_documents.append(fileids[x])
			test_categories.append(category)
	trains[category]=fileids_train
	tests[category]=fileids_test

def unigram(arrayOfWords):
	temp={}
	for word in arrayOfWords:
		w=stemmer.stem(word.lower())
		if (w not in stop_words) and (w not in string.punctuation) and (w not in ["''","``","--"]):
			if w in temp.keys():
				temp[w]+=1
			else:
				temp[w]=1
	return temp

def trainMultinominalNB(C,D,ngram):
	V=[]
	nonstem=[]
	for document in D:
		for word in movie_reviews.words(fileids=[document]):
			if (word.lower() not in nonstem) and (word.lower() not in stop_words) and (word.lower() not in string.punctuation) and (word.lower() not in ["''","``","--"]):
					nonstem.append(word.lower())
	for word in nonstem:
		if stemmer.stem(word) not in V:
			V.append(stemmer.stem(word))
	N=len(D)
	condprob={}
	prior={}
	NC={}
	T={}
	sumTct={}
	textc={}
	for category in C:
		NC[category]=len(trains[category]) #num of documents in D that below to class c
		prior[category]=NC[category]/N
		textc[category]=movie_reviews.words(fileids=trains[category])
		if ngram=="unigram":
			tokenCount=unigram(textc[category])
			T[category]={}
			for t in V:
				if t in tokenCount:
					T[category][t]=tokenCount[t]
				else:
					T[category][t]=0
			sumTct[category]=0
			for t in V:
				sumTct[category]+=(T[category][t]+1)

			for t in V:
				condprob[t]={}
				condprob[t][category]=(T[category][t]+1)/sumTct[category]
		else:
			bigramCounts=bigram(textc[category])
			unigramCounts=unigram(textc[category])
			vocabulary=vocab(textc[category])
			#this sequence of terms appears x amount of times in this category->bigramcounts for this category
			bigramProb={}
			for word1 in bigramCounts.keys():
				for word2 in bigramCounts[word1].keys():
					if bigramCounts[word1][word2] > 5:
						if word1 not in bigramProb.keys():
							bigramProb[word1]={}
						if word2 not in bigramProb[word1].keys():
							bigramProb[word1][word2]=0.0
						bigramProb[word1][word2]=(bigramCounts[word1][word2]+1)/(unigramCounts[word1]+vocabulary)
						condprob[word1]={}
						condprob[word1][word2]={}
						condprob[word1][word2][category]=bigramProb[word1][word2]
	return V, prior, condprob

def applymutinominalNB(C,V,prior,condprob,d,ngram):
	W=[]
	for w in d:
		if stemmer.stem(w) in V:
			W.append(stemmer.stem(w))
	score={}
	for category in C:
		score[category]=math.log(prior[category])
		if ngram=="unigram":
			for t in W:
				if category in condprob[t].keys():
					score[category]=math.log(condprob[t][category])
		else:
			for x in range(0,len(d)-1):
				word1=stemmer.stem(d[x])
				word2=stemmer.stem(d[x+1])
				if word1 in condprob.keys():
					if word2 in condprob[word1].keys():
						if category in condprob[word1][word2].keys():
							score[category]*=math.log(condprob[word1][word2][category])
	v=list(score.values())
	k=list(score.keys())
	maxy=max(v)
	max_elements=[]
	for category in score.keys():
		if score[category]==maxy:
			max_elements.append(category)
	return max_elements
	#return [k[v.index(max(v))]]


def normalMultinominalNB(V,prior,condprob,ngram):
	results=[]
	for doc in test_documents:
		d=movie_reviews.words(fileids=[doc])
		choice=applymutinominalNB(categories,V,prior,condprob,d,ngram)
		results.append(choice)
	right=0
	wrong=0
	for x in range(0,len(results)):
		if test_categories[x] in results[x]:
			right+=1
		else:
			wrong+=1

	right=right/len(results)
	wrong=wrong/len(results)

	print("Accuracy of "+ngram+" : "+str(right*100))
	print("Inaccuracy of "+ngram+" : "+str(wrong*100))

def bigram (wordList):
	bigramCount={}
	for x in range(0,len(wordList)-1):
		word1=stemmer.stem(wordList[x].lower())
		word2=stemmer.stem(wordList[x+1].lower())
		if (word1 not in stop_words) and (word1 not in string.punctuation) and (word1 not in ["''","``","--"]):
			if (word2 not in stop_words) and (word2 not in string.punctuation) and (word2 not in ["''","``","--"]):
				if word1 not in bigramCount.keys():
					bigramCount[word1]={}
				if word2 not in bigramCount[word1].keys():
					bigramCount[word1][word2]=1
				else:
					bigramCount[word1][word2]+=1
	return bigramCount;

def vocab (wordList):
	vocab=[]
	for word in wordList:
		if (word.lower() not in stop_words) and (word.lower() not in string.punctuation) and (word.lower() not in ["''","``","--"]):
			if stemmer.stem(word.lower()) not in vocab:
				vocab.append(stemmer.stem(word.lower()))
	return len(vocab)

V,prior,condprob=trainMultinominalNB(categories,train_documents,"unigram")
normalMultinominalNB(V,prior,condprob,"unigram")
V,prior,condprob=trainMultinominalNB(categories,train_documents,"bigram")
normalMultinominalNB(V,prior,condprob,"bigram")