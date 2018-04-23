#convert all counts to wordcount/totalwordcount
for category in bag_of_words.keys():
	totalCount=0;
	#get total number of words in current category
	for word in bag_of_words[category].keys():
		totalCount+=bag_of_words[category][word]
	for word in bag_of_words[category].keys():
		bag_of_words[category][word]=bag_of_words[category][word]/totalCount