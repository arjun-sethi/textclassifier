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

to_delete=[]
for word in inverse_frequency.keys():
	inverse_frequency[word]=math.log((len(bag_of_words.keys())/inverse_frequency[word]))
	if inverse_frequency[word]==0.0: #if the  word appears in all categories
		to_delete.append(word)

#calculate termfreq*inversefreq
for category in bag_of_words.keys():
	for word in to_delete:
		del bag_of_words[category][word] #delete the word from every category
		#mutliply term frequency by inverse term frequency
		#bag_of_words[category][word]=bag_of_words[category][word]*inverse_frequency[word]

totalCount={}
#convert all counts to wordcount/totalwordcount
for category in bag_of_words.keys():
	totalCount[category]=0;
	#get total number of words in current category
	for word in bag_of_words[category].keys():
		totalCount[category]+=bag_of_words[category][word]
	for word in bag_of_words[category].keys():
		bag_of_words[category][word]=bag_of_words[category][word]/totalCount[category]


#write word : frequency to file, named by category
for category in bag_of_words.keys():
	file=open(category+".txt","w")
	for word in bag_of_words[category].keys():
		file.write(word+" :\t"+str(bag_of_words[category][word])+"\n")
	file.close()

##POS TAGGING COUNTS
#testFile=[]
testBags={}
for category in categories:
	testBags[category]={}
	for word in brown.words(fileids=tests[category]):
		if word.lower() in testBags[category].keys():
			testBags[category][word.lower()]+=1
		else:
			testBags[category][word.lower()]=1

probability ={} #probabily of a given category being in categoryTest

for category in testBags.keys(): #for every category we are testing
	probability[category]={}
	for categoryTest in testBags.keys():
		probability[category][categoryTest]=1.000000 #create an instace of each category probabilyt for the category being tested
		for word in testBags[category].keys(): #for every word that appears in the test category
			if word in bag_of_words[categoryTest].keys(): #check if that word is in the trained set for every category
				if bag_of_words[categoryTest][word]==0.0:
					print("EQUAL TO ZERO: "+categoryTest+" "+word+" "+str(bag_of_words[categoryTest][word]))
				probability[category][categoryTest]*=bag_of_words[categoryTest][word]

for category in probability.keys():
	for categoryTest in probability[category].keys():
		probability[category][categoryTest]=(1/15)*probability[category][categoryTest]*100
		if category=="news":
			print(categoryTest+" : "+str(probability[category][categoryTest]))