import os,sys
import numpy as np
import nltk
import re
import ipdb

data_dir = "../datasets/semeval_2010_t1_eng/data"
#libs_dir = "hdp"
#sys.path.append(libs_dir)
from hdp.ugs import *

split_re = re.compile(r'([()|])')
INF = 1000000

class Event:
	def __init__(self,_phrase='',_pos='NN',_priority=INF,_fin_phrase=-INF):
		self.phrase = _phrase
		self.pos = _pos
		self.priority = _priority
		self.entity_id = -1
		self.fin_phrase = _fin_phrase
		

	def __str__(self):
		res = "[ phrase:%s\n  pos: %s\n  priority: %i\n  entity_id: %i]" % (self.phrase,self.pos,self.priority,self.entity_id)
		return res

	def __repr__(self):
		res = "[ phrase:%s\n  pos: %s\n  priority: %i\n  entity_id: %i]" % (self.phrase,self.pos,self.priority,self.entity_id)
		return res


def getdata(filename):
	'''
	read data and preprocess
	@param filename: relative or absolute training filename
	return (documents: each doc is a list of tuples (entity,entity_id)
			vocabulary: list of entities as vocabulary
			)
	'''
	documents = []
	sentence = []
	vocabulary = set()

	for line in open(filename):
		line = line.strip('\n')
		if line.startswith('#begin'):
			doc = []
			sentence=[]
			event = Event()
			open_events = [event]
			continue
			#all temp files are reset
		if line.startswith('#end'):
			#save all temp docs
			doc = np.array(doc)
			documents.append(doc)
			continue
		
		if line!='':
			temp1 = line.split('\t')
			sentence.append(temp1)

		if line=='':
			# build tree
			nn = len(sentence)+1
			dep_graph = [[] for i in xrange(nn)]
			nodes_depth = np.zeros(nn)
			root = -1
			for comp in sentence:
				u = int(comp[0])
				v = int(comp[8])
				dep_graph[u].append(v)
				dep_graph[v].append(u)
				if v==0:
					root=v
			# assign height to nodes with BFS
			queue = [root]
			visited = np.zeros(nn)
			while len(queue)!=0:
				curr_node = queue.pop()
				if visited[curr_node]!=1:
					visited[curr_node]=1
					for v in dep_graph[curr_node]:
						if visited[v]!=1 and nodes_depth[v]<nodes_depth[curr_node]+1:
							nodes_depth[v]=nodes_depth[curr_node]+1
							queue.append(v)

			#print(open_events)
			#ipdb.set_trace()

			# Extract entities
			for comp in sentence:
				event = open_events[-1]
				n_tok = int(comp[0])
				token = comp[1]
				pos = comp[4]
				synt_head = nodes_depth[n_tok] # corrected height
				coref_str = comp[-1]

				# continue building phrase
				if pos[0]=='N' and pos==event.pos and n_tok==event.fin_phrase+1:
					if event.phrase!='':
						event.phrase+=' '
					event.phrase+=token
					event.priority = min(event.priority,synt_head)
					event.fin_phrase+=1
					open_events[-1] = event
				# update ENTITY
				elif pos in ['NN','NNS','NNP','NNPS','PRP','PRP$']:
					for i in range(len(open_events)):
						if synt_head<open_events[i].priority:
							# conserve ENTITY_ID
							new_event = Event(token,pos,synt_head,n_tok)
							new_event.entity_id = open_events[i].entity_id
							open_events[i] = new_event

				if coref_str=='_':
					continue

				# pos in ['NN','NNS','NNP','NNPS','PRP','PRP$'] and 
				temp = split_re.split(coref_str)
				splitted = [a for a in temp if a!='']
				k=0
				while(k<len(splitted)):		# | no hace nada
					if splitted[k]=='(':
						priority = INF
						if pos in ['NN','NNS','NNP','NNPS','PRP','PRP$']:
							priority = synt_head
						event = Event(token,pos,priority,n_tok)	# initialization
						open_events.append(event)
					elif splitted[k].isdigit():
						id = 0
						id = int(splitted[k])
						open_events[-1].entity_id = id # si es nuevo asigna, sino chanca el [mismo] id al ultimo activo
					elif splitted[k]==')':
						if open_events[-1].pos in ['NN','NNS','NNP','NNPS','PRP','PRP$']:
							doc.append(tuple([open_events[-1].phrase,open_events[-1].entity_id]) )
							vocabulary.add(open_events[-1].phrase)	# build vocab
						open_events.pop()
					k+=1
			#END-FOR-ENTITIES

			#reset sentence var
			sentence = []
		#END-IF-EMPTY-LINE
	#END-FOR-FILE
	documents = np.array(documents)
	vocabulary = list(vocabulary)
	return documents,vocabulary

def index_docs(documents,vocabulary):
	'''
	@param documents: each doc is a list of tuples (entity,id)
	@param vocabulary: list of entities' vocabulary
	return (docs in word_id format)
	'''
	N = documents.shape[0]
	new_documents = {}
	for i in xrange(N):
		doc = documents[i]
		new_doc = [vocab.index(ent) for ent in doc[:,0]]
		new_documents[i]=new_doc
	return new_documents


if __name__=="__main__":
	print("Reading training data...")
	data_file = os.path.join(data_dir,'en.train.txt')
	docs,vocab = getdata(data_file)
	print("Formating data...")
	formated_docs = index_docs(docs,vocab)

	print("Initializing HDP...")
	gs = UncollapsedGibbsSampling(50);
	gs._initialize(formated_docs,gamma=1.0,alpha=1.0,eta=0.01)

	print("Sampling HDP (training)...")
	gs.sample(iteration=200,directory='output/');

