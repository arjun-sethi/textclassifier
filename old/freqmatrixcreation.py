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

tfidf = TfidfTransformer()
fitted=tfidf.fit(freq_term_matrix)

file.write("\n")
x=0
for category in bag_of_words.keys():
	file.write(category+"\t")
	for word in array:
		file.write(str(freq_term_matrix[x][allwords.index(word)])+"\t")
	file.write("\n")
	x+=1
file.close()