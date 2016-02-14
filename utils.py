import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import ipdb
import glob as gb
import codecs

np.seterr(divide='ignore')

split_re = re.compile(r'([()|])')

class Event:
	def __init__(self,_phrase='',_pos='NN',_priority=-np.inf,_fin_phrase=-np.inf):
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

'''
read corefenrence entities and preprocess
@param filename: absolute path + filename
return
	(documents: [doc| doc=(entity,POS,entity_id)]
	 vocabulary: list of entities as vocabulary
	)
'''
def read_conll2010_task1(filename,n_docs='all'):
	documents = []
	sentence = []
	vocabulary = set()
	count_docs = 0

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
			count_docs+=1
			if n_docs!='all':
				if count_docs>=n_docs:
					break
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

			# Extract entities
			for comp in sentence:
				event = open_events[-1]
				n_tok = int(comp[0])
				token = comp[1]
				pos = comp[4]
				synt_head = nodes_depth[n_tok] # corrected height
				coref_str = comp[-1]

				if pos not in ['NNP','NNPS'] and token!="I": # lowercase any other than proper nouns
					token = token.lower()

				for i in xrange(len(open_events)):
					event = open_events[i]
					# continue building phrase
					if pos[0]=='N' and pos==event.pos and n_tok==event.fin_phrase+1:
						if event.phrase!='':
							event.phrase+=' '
						event.phrase+=token
						event.priority = min(event.priority,synt_head)
						event.fin_phrase+=1
						open_events[i] = event
					# update ENTITY
					elif pos in ['NN','NNS','NNP','NNPS','PRP','PRP$'] and synt_head<event.priority:
							# conserve ENTITY_ID
							new_event = Event(token,pos,synt_head,n_tok)
							new_event.entity_id = event.entity_id
							open_events[i] = new_event

				if coref_str=='_':
					continue

				# pos in ['NN','NNS','NNP','NNPS','PRP','PRP$'] and 
				temp = split_re.split(coref_str)
				splitted = [a for a in temp if a!='']
				k=0
				while(k<len(splitted)):		# | no hace nada
					if splitted[k]=='(':
						priority = np.inf
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
							text = open_events[-1].phrase
							ev_pos = open_events[-1].pos
							ent_id = open_events[-1].entity_id
							doc.append(tuple([text,ev_pos,ent_id]) )
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



def read_noun_lexicon(_dir):
	nouns_vocab = set()
	rep = 0
	for filename in gb.glob(_dir+'*'):
		print(filename)
		k=0
		last_line=""
		try:
			for line in open(filename,encoding='latin-1'):
				k+=1
				line = line.strip("\n")
				last_line=line
				if line=='':
					continue
				splitted = line.split("\t")
				np = splitted[0]
		
				try:
					mas,fem,neu,plu = [int(a) for a in splitted[1].split(" ")]
				except:
					ipdb.set_trace()
				if np in nouns_vocab:
					rep+=1
		except:
			print(k)
			print(last_line)
	return rep


