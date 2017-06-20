#-*- coding: UTF-8 -*-  

#import the necessary packages
import numpy
import copy
import theano
import random

#define the function(get the data)
def genBatch(data):
    m =0 

	#the length of the data
    maxsentencenum = len(data[0])

	#through the all data, "doc" is each traverse to the item
    for doc in data:
		
		#through each
        for sentence in doc:

			#judge! if the length > m
            if len(sentence)>m:
                m = len(sentence)

        for i in xrange(maxsentencenum - len(doc)):
			
			#into the stack("doc")
            doc.append([-1])
	
	#map
    tmp = map(lambda doc: numpy.asarray(map(lambda sentence : sentence + [-1]*(m - len(sentence)), doc), dtype = numpy.int32).T, data)                          #[-1]是加在最前面
    
	#reduce
	tmp = reduce(lambda doc,docs : numpy.concatenate((doc,docs),axis = 1),tmp)
    return tmp 

#get the length
def genLenBatch(lengths,maxsentencenum):
    lengths = map(lambda length : numpy.asarray(length + [1.0]*(maxsentencenum-len(length)), dtype = numpy.float32)+numpy.float32(1e-4),lengths)
    return reduce(lambda x,y : numpy.concatenate((x,y),axis = 0),lengths)

#get the result matrix
def genwordmask(docsbatch):
    mask = copy.deepcopy(docsbatch)
    mask = map(lambda x : map(lambda y : [1.0 ,0.0][y == -1],x), mask)
    mask = numpy.asarray(mask,dtype=numpy.float32)
    return mask

#get the transposed matrix
def gensentencemask(sentencenum):
    maxnum = sentencenum[0]
    mask = numpy.asarray(map(lambda num : [1.0]*num + [0.0]*(maxnum - num),sentencenum), dtype = numpy.float32)
    return mask.T

#the main class for dataset
class Dataset(object):
	
	#when an instance，initialize
    def __init__(self, filename, emb,maxbatch = 32,maxword = 500):
        lines = map(lambda x: x.split('\t\t'), open(filename).readlines())           
        label = numpy.asarray(
            map(lambda x: int(x[2])-1, lines),
            dtype = numpy.int32
        )
        docs = map(lambda x: x[3][0:len(x[3])-1], lines) 
        docs = map(lambda x: x.split('<sssss>'), docs) 
        docs = map(lambda doc: map(lambda sentence: sentence.split(' '),doc),docs)
        docs = map(lambda doc: map(lambda sentence: filter(lambda wordid: wordid !=-1,map(lambda word: emb.getID(word),sentence)),doc),docs)
        tmp = zip(docs, label)
        #random.shuffle(tmp)
        tmp.sort(lambda x, y: len(y[0]) - len(x[0]))  
        docs, label = zip(*tmp)

        sentencenum = map(lambda x : len(x),docs)
        length = map(lambda doc : map(lambda sentence : len(sentence), doc), docs)
        self.epoch = len(docs) / maxbatch                                      
        if len(docs) % maxbatch != 0:
            self.epoch += 1
    
		#the [] is a empty list(the built-in stack in Python)
		#func: append (into the stack)
		#func: pop (out of the stack)
        self.docs = []
        self.label = []
        self.length = []
        self.sentencenum = []
        self.wordmask = []
        self.sentencemask = []
        self.maxsentencenum = []

		#through the self.epoch
        for i in xrange(self.epoch):
            self.maxsentencenum.append(sentencenum[i*maxbatch])
            self.length.append(genLenBatch(length[i*maxbatch:(i+1)*maxbatch],sentencenum[i*maxbatch])) 
            docsbatch = genBatch(docs[i*maxbatch:(i+1)*maxbatch])
            self.docs.append(docsbatch)
            self.label.append(numpy.asarray(label[i*maxbatch:(i+1)*maxbatch], dtype = numpy.int32))
            self.sentencenum.append(numpy.asarray(sentencenum[i*maxbatch:(i+1)*maxbatch],dtype = numpy.float32)+numpy.float32(1e-4))
            self.wordmask.append(genwordmask(docsbatch))
            self.sentencemask.append(gensentencemask(sentencenum[i*maxbatch:(i+1)*maxbatch]))
        

class Wordlist(object):
    def __init__(self, filename, maxn = 100000):

		#open(filename): get the file object
		#readlines(): the return value is a list(or a stack), each item is a line of files
		#[:maxn]: get the items from 0 to maxn
        lines = map(lambda x: x.split(), open(filename).readlines()[:maxn])
        
		#get the rows number of the file
		self.size = len(lines)

        self.voc = [(item[0][0], item[1]) for item in zip(lines, xrange(self.size))]

		#the dictionary
        self.voc = dict(self.voc)

	#the function to get the ID of the word
    def getID(self, word):
		
		#try-except to capture the Exception
        try:
            return self.voc[word]
        except:
			
			#return the -1(failure)
            return -1
