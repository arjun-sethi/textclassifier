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
		wordsUnsani=brown.words(fileids=[trains[category][x]])
		wordssani=[]
		for y in range(0,len(wordsUnsani)):
			wordssani.append(stemmer.stem(wordsUnsani[y]))
		mystring=' '.join(wordssani)
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
		wordsUnsani=brown.words(fileids=[tests[category][x]])
		wordssani=[]
		for y in range(0,len(wordsUnsani)):
			wordssani.append(stemmer.stem(wordsUnsani[y]))
		mystring=' '.join(wordssani)
		dataTest.append(mystring.translate(str.maketrans('','',string.punctuation)))
		targetsTest.append(category)
predicted = text_clf.predict(dataTest)
print(len(dataTest))
np.mean(predicted == targetsTest)
print(np.mean(predicted == targetsTest)*100)