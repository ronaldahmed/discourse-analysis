import os,sys
import numpy as np
from utils_topicality import *
import pickle
import ipdb

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

	print("Saving HDP model...")
	with open('HDP.pickle', 'wb') as fd:
		pickle.dump(gs, fd, protocol=pickle.HIGHEST_PROTOCOL)