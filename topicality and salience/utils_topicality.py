import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import ipdb

sys.path.append("../")
from utils import *
from hdp.ugs import *

numpy.seterr(divide='ignore')

'''
@param refs_by_doc: [doc | doc:{"lex":{ref:freq}, "pro":{ref:freq} }]
return ( docs: {id | doc[id]:[ref_id]}
		 refs_vocab: vocabulary of only referents)
'''
def index_docs(documents):
	N = len(documents)
	new_documents = {}
	refs_vocab = []
	for i in xrange(N):
		doc = documents[i]
		new_doc = []
		for ref,freq in doc["lex"].items():
			ref_id = -1
			if ref not in refs_vocab:
				ref_id = len(refs_vocab)
				refs_vocab.append(ref)
			else:
				ref_id = refs_vocab.index(ref)
			new_doc += freq*[ref_id]

		for ref,freq in doc["pro"].items():
			ref_id = -1
			if ref not in refs_vocab:
				ref_id = len(refs_vocab)
				refs_vocab.append(ref)
			else:
				ref_id = refs_vocab.index(ref)
			new_doc += freq*[ref_id]
		new_documents[i]=new_doc
	return new_documents,refs_vocab

def add(_dict,key,val):
	if key not in _dict:
		_dict[key]=0
	_dict[key]+=val
	return

"""
Extracts referent chains, assigns the referent and separate by lex or pro, + frequency
return
	refs_by_doc: [doc | doc:{"lex":{ref:freq}, "pro":{ref:freq} }]
	ref_by_token: {(pos_weight,token):[referents]}
"""
def get_referents(documents):
	N = documents.shape[0]

	ref_by_token = {}
	pos_weights = {
		'NNP':0,
		'NNPS':0,
		'NN':5,
		'NNS':5,
		'PRP':10,
		'PRP$':10
	}
	refs_by_doc = []
	for i in xrange(N):
		chains = {}
		lexical_noun_refs = {}
		pronoun_refs = {}

		# get annotated correference chains
		for (ent,pos,ent_id) in documents[i]:
			if ent_id not in chains:
				chains[ent_id] = set()
			if (ent,pos) not in chains[ent_id]:
				chains[ent_id].add((pos_weights[pos],ent))
		
		# get lexical nouns and pronouns referents
		for (ent_id,ref_chain) in chains.items():
			chain = set(ref_chain)
			(pos_w,ref) = chain.pop()
			if pos_w==10:	# Only Nouns as referents
				continue
			# add to lexical_noun referents
			add(lexical_noun_refs,ref,1)
			# map token to referent
			token = ref
			if (pos_w,token) not in ref_by_token:
				ref_by_token[(pos_w,token)] = set()
			ref_by_token[(pos_w,token)].add(ref)

			while(len(chain)>0):
				pos_w,token = chain.pop()
				if pos_w==10:
					add(pronoun_refs,ref,1)
				else:
					add(lexical_noun_refs,ref,1)
				# map token to referent
				if (pos_w,token) not in ref_by_token:
					ref_by_token[(pos_w,token)] = set()
				ref_by_token[(pos_w,token)].add(ref)

		# save by doc
		doc = {}
		doc["lex"] = lexical_noun_refs
		doc["pro"] = pronoun_refs
		refs_by_doc.append(doc)
	return refs_by_doc,ref_by_token
		
"""
Assigns k from argmax p(xij|k,t). p(z_{ji}|G_j) = gamma * Gamma(gamma) * Gamma(m_{.k}) * fact(gamma-1) / fact(m_{..}+gamma-1)
"""
def log_pz(hdp,document_index,word,vocab):
	word_index = hdp._vocab.index(vocab.index(word))
	max_topic_index = hdp.assign_topic_word(word_index)

	if max_topic_index==-1:
		return -np.inf
		
	# p(t_{ij}=t)
	"""
	table_log_likelihood = len(hdp._k_dt[document_index]) * np.log(hdp._alpha) - log_factorial(len(hdp._t_dv[document_index]), hdp._alpha);
	for table_index in xrange(len(hdp._k_dt[document_index])):
	    table_log_likelihood += scipy.special.gammaln(hdp._n_dt[document_index][table_index]);
	table_log_likelihood += scipy.special.gammaln(hdp._alpha);
	"""

	# p(k_{jt}=z_{ji}|t_{ji})
	topic_log_likelihood = np.log(hdp._gamma) - log_factorial(np.sum(hdp._m_k), hdp._gamma)
	topic_log_likelihood += scipy.special.gammaln(hdp._m_k[max_topic_index]);
	topic_log_likelihood += scipy.special.gammaln(hdp._gamma);

	return  topic_log_likelihood #* table_log_likelihood


"""
p(z_{ji}|G_j) = n_{j.k} / Sum(n_{j..})
"""
def log_pz_ml(hdp,document_index,word,vocab):
	word_index = hdp._vocab.index(vocab.index(word))
	log_pz = np.log(hdp._n_kd[:,document_index]) - np.log(np.sum(hdp._n_kd[:,document_index]))
	k = log_pz.argmax()
	return log_pz.max()

def expectation_ref(refs,hdp,doc_idx,vocab):
	E_log_pz = 0
	n_refs = 0
	for ref,freq in refs.items():
		n_refs += freq
		E_log_pz += freq*log_pz(hdp,doc_idx,ref,vocab)
	if n_refs==0:
		return -np.inf
	E_log_pz /= n_refs
	return E_log_pz

def expectation_ref_counts(refs,doc):
	E_log_pz = 0
	n_total = sum(doc["lex"].values())
	n_total += sum(doc["pro"].values())
	n_refs = 0
	
	for ref,freq in refs.items():
		n_refs += freq
		E_log_pz += np.log(freq) - np.log(n_total)		# p(ri) = C_{j,ri} / C_{j.}

	if n_refs==0:
		return -np.inf
	E_log_pz /= n_refs
	return E_log_pz

'''
@param refs_by_doc: [doc | doc:{"lex":{ref:freq}, "pro":{ref:freq} }]
return 
'''
def get_lr_format(refs_by_doc,hdp,refs_vocab):
	# model 1: only p_ri
	# model 2: p_ri; p(z|t,k)
	X = []
	Y = []
	N = len(refs_by_doc)
	for i in xrange(N):
		doc = refs_by_doc[i]
		c_d = sum(doc["lex"].values())
		c_d += sum(doc["pro"].values())
		for ref,freq in doc["lex"].items():
			log_pri = np.log(freq) - np.log(c_d)
			log_pz_ref = log_pz(hdp,i,ref,refs_vocab)

			X.append([log_pri,log_pz_ref])
			Y.append(0)

		for ref,freq in doc["pro"].items():
			log_pri = np.log(freq) - np.log(c_d)
			log_pz_ref = log_pz(hdp,i,ref,refs_vocab)

			X.append([log_pri,log_pz_ref])
			Y.append(1)
	return (X,Y)


