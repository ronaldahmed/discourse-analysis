import os,sys
import numpy as np
import glob as gb
import re
import ipdb
import pickle
import copy

np.seterr(divide='ignore')
split_re = re.compile(r'([()|])')
###################################################################################

reflexives = {
	'myself',
	'yourself',
	'himself',
	'herself',
	'itself',
	'yourselves',
	'themselves',
	'ourselves',
}

indexicals = {	# not present in OntoNotes
	'that',
	'this',
}

dep_rels_considered = [
	'SBJ',
	'OBJ',
	'PMOD',
]

third_person_prp = [
	'he',
	'him',
	'his',		# always NMOD, so never extracted
	'she',
	'her',
	'hers',
	'it',
	'its',		# always NMOD or COORD, so never extracted
	'they',
	'them',
	'their',	# always NMOD, so never extracted
]

pronounBydeprel = {
	'SBJ': ['he','him','she','her','hers','it','they'],
	'OBJ': ['him','her','it','them','they'],
	'PMOD': ['him','her','it','them']
}

pronoun_type_debug = [
	'sing_mas',
	'sing_fem',
	'sing_neu',
	'plural',
	'<NOT FOUND>',
]

pronounBytype = {
	'sing_mas': ['he','him','his'],
	'sing_fem': ['she','her','hers'],
	'sing_neu': ['it','its'],
	'plural': ['they','them','their']
}

#################################################################################################
class Entity:
	def __init__(self,_phrase='',_pos='-',global_pos=-np.inf):
		self.phrase = _phrase
		self.pos = _pos
		self.global_pos = global_pos
		self.entity_id = -1
		self.referent_id = -1 # global id in referents vocabulary
		self.dep_rel = ""

		
	def __str__(self):
		gp = self.global_pos
		if gp==-np.inf:
			gp=-1
		res = "[ phrase:%s |  pos: %s |  global_pos: %i |  entity_id: %i  | dep_rel: %s ]" % (self.phrase,self.pos,gp,self.entity_id,self.dep_rel)
		return res

	def __repr__(self):
		gp = self.global_pos
		if gp==-np.inf:
			gp=-1
		res = "[ phrase:%s |  pos: %s |  global_pos: %i |  entity_id: %i  | dep_rel: %s ]" % (self.phrase,self.pos,gp,self.entity_id,self.dep_rel)
		return res

"""
Read entities, considering maximally spanning NPs (ignoring nested NPs and nested entities).
Entity selection is done with constraints in NNP and PRP - see paper -.

"""
def read_conll2010_task1_max_np(filename,n_docs='all'):
	documents = []
	sentence = []
	count_docs = 0
	nesting_depth = np.inf
	global_line = 0

	for line in open(filename):
		line = line.strip('\n')
		if line.startswith('#begin'):
			doc = []
			entity_filter = np.zeros(1000,dtype=bool)	# {ent_id: True, if there is at least one NNP in the chain}
			chain_lens = np.zeros(1000)	# {ent_id: count of elements in the chain}

			nesting_depth = np.inf
			event = Entity()
			global_line = 0
			continue
			#all temp files are reset
		if line.startswith('#end'):
			documents.append(doc)
			count_docs+=1
			if n_docs!='all':
				if count_docs>=n_docs:
					break
			continue
		
		if line=='':
			nesting_depth = np.inf	# default val=INF
			continue

		## MAIN STUFF
		global_line += 1 		# increment in each valid line
		comp = line.split('\t')
		token = comp[1].lower()		# consider lowercase for all
		pos = comp[4]
		dep_rel = comp[10]
		coref_str = comp[-1]

		"""
		# only consecutive proper nouns
		if nesting_depth!=np.inf:
			if pos in ['NNP','NNPS'] and event.pos==pos:
				event.phrase += " "+token
			else:
				event.pos = '-'
		"""
		
		# maximally spanning NPs considered, labeled as NNP if there is an NNP in the phrase
		if nesting_depth!=np.inf:	# only enters for maximally entities
			event.phrase += " "+token
			if pos in ['NNP','NNPS']:
				event.pos = pos
			elif event.pos not in ['NNP','NNPS']:
				event.pos = '-'

			if dep_rel=='SBJ' and event.dep_rel!='SBJ':
				event.dep_rel = dep_rel
			elif dep_rel=='OBJ' and event.dep_rel!='SBJ':
				event.dep_rel = dep_rel
			elif dep_rel=='PMOD' and event.dep_rel not in ['SBJ','OBJ','NAME']:
				event.dep_rel = dep_rel

		if coref_str=='_':
			continue

		temp = split_re.split(coref_str)
		splitted = [a for a in temp if a!='']
		k=0

		while(k<len(splitted)):		# | no hace nada
			if splitted[k]=='(':
				if nesting_depth==np.inf:
					nesting_depth=0
					event = Entity(token,pos,global_line)	# initialization of active entity
					event.dep_rel = dep_rel
				nesting_depth+=1
			elif splitted[k].isdigit():
				id = int(splitted[k])
				if nesting_depth==1:	# depth of 1 corresponds to maximal entity
					event.entity_id = id
			elif splitted[k]==')':
				nesting_depth-=1
				if nesting_depth==0: 	# if ) is closing maximal
					# cleaning phrase
					phrase = event.phrase.strip(' ')
					if len(phrase)>1:
						if phrase[-2:]==' .' or phrase[-2:]==' ,':	# if ends in point or comma
							phrase = phrase[:-2]
						phrase = phrase.strip(' ')	# get rid of spaces on sides
					event.phrase = phrase

					if any([ event.pos in ['NNP','NNPS'], 
							 all([	event.pos in ['PRP','PRP$'],		# all constraints in selection of pronouns
							 		token in third_person_prp,		# third person pronouns
							 		token not in reflexives,		# no reflexives
							 		token not in indexicals,		# no indexicals
							 		event.dep_rel in dep_rels_considered,	# pronouns in SUBJ, OBJ or PMOD position
							 	])
						]):
						# filtering preprocessing
						if event.pos in ['NNP','NNPS']:				# found an NNP in the chain
							entity_filter[event.entity_id] = True
							if event.dep_rel not in dep_rels_considered:
								event.dep_rel = 'SBJ'
						doc.append(event)

					nesting_depth=np.inf
			k+=1
	#END-FOR-FILE

	documents = np.array(documents)
	
	return documents

"""
Filter entities according to the following rules:
 - at least one NNP in chain
 - if one NNP in chain is present in the lexicon, consider all NNPs of chain
"""
def filter_entities(documents,lexicon):
	filtered_docs = []
	for doc in documents:
		# all elements in chains with at least one valid NNP will be considered
		valid_chains = set()
		for entity in doc:
			if any([ entity.pos=='NNPS',								#plural goes in anyway
					 entity.pos=='NNP' and entity.phrase in lexicon,	# only if present in lexicon vocabulary
					]):
				valid_chains.add(entity.entity_id)
		# filter documents
		new_doc = np.array( [entity for entity in doc if entity.entity_id in valid_chains ] )
		if len(new_doc)!=0:
			filtered_docs.append(new_doc)

	filtered_docs = np.array(filtered_docs)
	return filtered_docs	


#################################################################################################
"""
Read NP lexicon with genre and number counts. Only considered the cases: (\w )+, (! \w), (\w !)
return:
	counts: (V,4) : [masculine_count  femenine_count  neutral_count  plural_count]
	nouns_vocab (V,) : vocabulary of NP
"""
def read_noun_lexicon(_dir):
	nouns_vocab = []
	counts = []
	for filename in gb.glob(_dir+'*'):
		for line in open(filename,encoding='latin-1'):
			line = line.strip("\n")
			last_line=line
			if line=='':
				continue
			splitted = line.split("\t")
			phrase = splitted[0]

			mas,fem,neu,plu = [int(a) for a in splitted[1].split(" ")]
			sp_np = phrase.split(' ')

			save = False
			if all([word.isalpha() for word in sp_np]):
				save = True
			"""
			# save <! word> or <word !>
			if len(sp_np)>1:
				if any([
					 len(sp_np)>1 and (sp_np[0]=='!' and sp_np[1].isalpha()),
					 len(sp_np)>1 and (sp_np[1]=='!' and sp_np[0].isalpha()),
				]):
					save = True
			"""
			if save:
				counts.append([mas,fem,neu,plu])
				nouns_vocab.append(phrase)
	counts = np.array(counts)
	nouns_vocab = np.array(nouns_vocab)
	return counts,nouns_vocab

"""
Get intersection between corpus vocabulary and genre_number lexicon vocabulary
"""
def get_vocab_from_lexicon(documents,lexicon_vocab):
	vocab = set()
	for doc in documents:
		for ent in doc:
			if ent.phrase in lexicon_vocab:
				vocab.add(ent.phrase)
	return vocab



"""
Annotate each phrase in vocab with genre and number (pronoun type id)
"""
def annotate_pro_type(lex_counts,lex_vocab,corpus_vocab):
	pro_type_ids = []
	for phrase in corpus_vocab:
		pro_id=-1
		if phrase in lex_vocab:
			p_idx = np.nonzero(lex_vocab==phrase)[0][0]
			pro_id = lex_counts[p_idx,:].argmax()
		elif phrase in third_person_prp:
			for _type,pros in pronounBytype.items():
				if phrase in pros:
					pro_id = pronoun_type_debug.index(_type)
					break
		else:
			pro_id = pronoun_type_debug.index('plural')		# if not in lexicon, PLURAL, since all NNPs are considered
		pro_type_ids.append(pro_id)
	pro_type_ids = np.array(pro_type_ids)
	return pro_type_ids

def saveObject(obj, name='model'):
	with open(name + '.pickle', 'wb') as fd:
		pickle.dump(obj, fd, protocol=pickle.HIGHEST_PROTOCOL)

def uploadObject(obj_name):
	# Load tagger
	with open(obj_name + '.pickle', 'rb') as fd:
		obj = pickle.load(fd)
	return obj

#################################################################################################
"""
Get orreference chains from a single document
@param doc: list of Entity objects
"""
def get_coref_chains(doc):
	chains = {}
	for entity in doc:
		ent_id = entity.entity_id
		if ent_id not in chains:
			chains[ent_id] = []
		pair = tuple([entity.phrase,entity.pos])
		if pair not in chains[ent_id]:
			chains[ent_id].append(pair)
	
	return chains

def get_referents(doc,lexicon):
	referents = {}	# 1000 assumed to be max ent_id
	chains = get_coref_chains(doc)

	for chain_id,chain in chains.items():
		for ref_exp in chain:
			text = ref_exp[0]
			pos = ref_exp[1]
			if pos=='NNP' and text in lexicon:
				referents[chain_id] = text
			elif pos=='NNPS':
				referents[chain_id] = text
				break
	return referents

'''
Reformat corpus and get vocabulary of refering expressions (pronouns and referents).
return: new_documents: [[entity_obj with phrase:ref_exp_id, and entity_id]]
		ref_exp_vocab: [all referents + all pronouns (third person PRP)]  -> assumption: a proper name only refers to itself
'''
def reformat_data(documents,lexicon):
	ref_exp_vocab=[]
	# make referents vocabulary and coreference chains
	for doc in documents:
		referents = get_referents(doc,lexicon)
		for entity in doc:
			referent = referents[entity.entity_id]
			ref_id=-1
			if referent in ref_exp_vocab:
				ref_id = ref_exp_vocab.index(referent)
			else:
				ref_id = len(ref_exp_vocab)
				ref_exp_vocab.append(referent)
			entity.referent_id = ref_id
			if entity.pos[0]=='N':			# replace original noun with referent (assumption: proper names only refer to themselves)
				entity.phrase = referent
	# add pronouns
	for pro in third_person_prp:
		if pro not in ref_exp_vocab:
			ref_exp_vocab.append(pro)

	# reformat documents
	new_documents = []
	for doc in documents:
		new_doc = copy.deepcopy(doc)
		for event in new_doc:
			event.phrase = ref_exp_vocab.index(event.phrase)
		new_documents.append(new_doc)
	new_documents = np.array(new_documents)
	ref_exp_vocab = np.array(ref_exp_vocab)

	return new_documents,ref_exp_vocab

"""
Change SpeakerModel format to HDP_UGS format
return: {doc_id: [word_id....]}
"""
def reformat_hdp(documents):
	hdp_format = {}
	for i,doc in enumerate(documents):
		new_doc = [entity.referent_id for entity in doc]
		hdp_format[i] = new_doc
	return hdp_format


def debug_dep_rel_pronoun(data):
	debug = {
	'SBJ': set(),
	'OBJ': set(),
	'PMOD': set()
	}
	for doc in data:
		for ent in doc:
			if ent.pos[0]=='P':
				debug[ent.dep_rel].add(ent.phrase)
	return debug
