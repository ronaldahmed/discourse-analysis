import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils_topicality import *

data_dir = "../datasets/semeval_2010_t1_eng/data"

if __name__=="__main__":
	print("Reading training data...")
	n_docs='all'
	gibbs_iterations=1000
	data_file = os.path.join(data_dir,'en.train.txt')
	
	docs,vocab = read_conll2010_task1(data_file,n_docs=n_docs)
	
	# get referents by doc
	refs_by_doc,ref_by_token = get_referents(docs)

	print("Formating data...")
	formated_docs,referents_vocab = index_docs(refs_by_doc)

	print("Initializing HDP...")
	gs = UncollapsedGibbsSampling(50);
	gs._initialize(formated_docs,referents_vocab,gamma=1.0,alpha=1.0,eta=0.01)

	print("Sampling HDP (training)...")
	gs.sample(iteration=gibbs_iterations,directory='output/');
	
	print("\nTop n words by topics, ordered by p(k)...")
	gs.print_topics(referents_vocab,5)
	

	print("Calculating q_d...")	

	N = len(docs)
	e_lex = np.zeros(N)
	e_pro = np.zeros(N)
	# Calc E_pro and E_lex
	for doc_index in xrange(N):
		E_lex = expectation_ref(refs_by_doc[doc_index]["lex"],gs,doc_index,referents_vocab)
		E_pro = expectation_ref(refs_by_doc[doc_index]["pro"],gs,doc_index,referents_vocab)
		e_lex[doc_index] = E_lex
		e_pro[doc_index] = E_pro

	# Calc qd: topicalty measure
	qd = e_pro - e_lex
	# don't consider cases were e_pro==-INF
	noinf_idx = qd!=-np.inf
	# median test
	stat,p,m,_ = scipy.stats.median_test(e_pro[noinf_idx],e_lex[noinf_idx])
	print("\nMedia test for E_pro and E_lex: p=%.4f" % p)

	"""
	#p(q) as histogram
	plt.figure()
	plt.hist(qd[noinf_idx],20,normed=True,histtype='stepfilled')
	plt.title("p(q)")
	plt.grid(True)
	plt.show()
	"""
	
	# make: -pos_w,k_ord_idx,tuple_id
	target_ref_k = {} # pos_w: [k_ord,tupleid]
	token_ref_id = [] # token, ref as word_ids

	for ( (pos_w,token),refs ) in ref_by_token.items():
		for ref in refs:
			if ref==token:
				continue
			tr_idx = tuple([vocab.index(token),vocab.index(ref)])		# using global vocabulary
			tuple_id = -1
			if tr_idx in token_ref_id:
				tuple_id = token_ref_id.index(tr_idx)
			else:
				tuple_id = len(token_ref_id)
				token_ref_id.append(tr_idx)

			ref_id = gs._vocab.index(referents_vocab.index(ref))		# get referent id in HDP vocab
			k = gs.assign_topic_word(ref_id).argmax()
			k_ord_idx = np.nonzero(gs._ord_topics_pk[:,1]==k)[0][0]
			if pos_w not in target_ref_k:
				target_ref_k[pos_w] = []
			target_ref_k[pos_w].append( (k_ord_idx,tuple_id) )
	
	for key in target_ref_k.keys():
		target_ref_k[key].sort()

	## plot: target | referent | topic_id
	max_per_pos = 10
	print("\n%30s | %30s | ref k_order" % ("Target","Referent"))
	print("----------------------------------------------------------------------------")
	for pos,tups in target_ref_k.items():
		for i in xrange(max_per_pos):
			k_ord_idx ,tup_id = tups[i]
			target = vocab[ token_ref_id[tup_id][0] ]
			ref    = vocab[ token_ref_id[tup_id][1] ]
			#k = gs._ord_topics_pk[k_ord_idx,1]

			print("%30s | %30s | %i" % (target,ref,k_ord_idx))
		print("----------------------------------------------------------------------------")


	###########
	# Media test using prob referents directly from corpus
	print("\nCalculating q_d for p(ri|doc_d) from corpus...")	
	e_lex_corpus = np.zeros(N)
	e_pro_corpus = np.zeros(N)
	# Calc E_pro and E_lex
	for doc_index in xrange(N):
		E_lex = expectation_ref_counts(refs_by_doc[doc_index]["lex"],refs_by_doc[doc_index])
		E_pro = expectation_ref_counts(refs_by_doc[doc_index]["pro"],refs_by_doc[doc_index])
		e_lex_corpus[doc_index] = E_lex
		e_pro_corpus[doc_index] = E_pro

	# Calc qd: topicalty measure
	qd_corpus = e_pro_corpus - e_lex_corpus
	# don't consider cases were e_pro==-INF
	noinf_idx = qd_corpus!=-np.inf
	# median test
	stat,p,m,_ = scipy.stats.median_test(e_pro_corpus[noinf_idx],e_lex_corpus[noinf_idx])
	print("\nMedia test for E_pro and E_lex: p=%.4f" % p)

	# get features for models 1 and 2 and export to work in R
	x,y = get_lr_format(refs_by_doc,gs,referents_vocab)
	output = open("lr_models.csv",'w')
	output.write("Y,X1,X2\n")
	for i in xrange(len(y)):
		output.write("%i,%.4f,%.4f\n" % (y[i],x[i][0],x[i][1]))
	
	ipdb.set_trace()

