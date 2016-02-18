import os,sys
import ipdb
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from class_speaker_model import SpeakerModel
from alternative_models import *

data_conll_dir = "../datasets/semeval_2010_t1_eng/data"
data_noun_lexicon_dir = "../datasets/noun_gender_number/"

if __name__=="__main__":
	n_docs='all'
	update_pron_types = False		# True: load agreement annotation for referents
	update_lexicon = False			# True: load vocabulary intersection between genre_number lexicon and corpus

	GN_counts = []
	GN_np_vocab = []

	# Read training data
	print("Reading training data...")
	data_file = os.path.join(data_conll_dir,'en.train.txt')
	docs = read_conll2010_task1_max_np(data_file,n_docs=n_docs)

	# read proper names present in genre_number lexicon, for further filtering
	gn_read = False
	if update_lexicon:
		print("Reading genre_number lexicon...")
		GN_counts, GN_np_vocab = read_noun_lexicon(data_noun_lexicon_dir)	# 1.9 GB of memory OMG
		gn_read = True
		lexicon = get_vocab_from_lexicon(docs,GN_np_vocab)
		saveObject(lexicon,'names_in_gnlexicon')
	else:
		lexicon = uploadObject('names_in_gnlexicon')

	
	docs = filter_entities(docs,lexicon)
	#deb = debug_dep_rel_pronoun(docs)
	
	# reformat data
	formated_data,vocab = reformat_data(docs,lexicon)

	# Visualizing Corpus dimensionality
	pros = 0
	nps = 0
	for doc in docs:
		nps  += sum([1 for ent in doc if ent.pos[0]=='N'])
		pros += sum([1 for ent in doc if ent.pos[0]=='P'])

	print("Pronouns: ",pros)		# curr: 557
	print("Proper nouns: ",nps)		# curr: 1233
	print("Ref_vocab: ",len(vocab))

	#ipdb.set_trace()

	# Getting annotation of genre and number for vocabulary
	pro_type_ids = []
	if update_pron_types:
		if not gn_read:
			print("Reading genre_number lexicon...")
			GN_counts, GN_np_vocab = read_noun_lexicon(data_noun_lexicon_dir)	# 1.9 GB of memory OMG
		pro_type_ids = annotate_pro_type(GN_counts,GN_np_vocab,vocab)
		
		n_pro_types = len(pronoun_type_debug)-1
		pro_type_counts = [sum(GN_counts[:,i]!=0) for i in range(n_pro_types)]
		pro_type_counts.append( GN_counts.shape[0] )

		saveObject(pro_type_ids,'pro_type_ids')
		saveObject(pro_type_counts,'pro_type_counts')
	else:
		pro_type_ids = uploadObject('pro_type_ids')
		pro_type_counts = uploadObject('pro_type_counts')

	
	print("Agreement annotations (pronoun type)")
	print("-"*150)
	for ii in range(len(pronoun_type_debug)-1):
		for i in range(20):
			if pro_type_ids[i]==ii:
				print("%120s : %15s" % (vocab[i],pronoun_type_debug[pro_type_ids[i]] ) )
		print("-"*150)
	
	tune_parameters = True		#True: tune alpha and recency decay
	if not tune_parameters:
		alpha = 1e-4
		decay = 100
		salience_measures = ['recency','frequency']
		print("Running models...")
		print("  alpha: ",alpha)
		print("  decay: ",decay)
		print("-"*150)

		print("%12s | %15s || %5s | %11s | %14s | %9s |" % ("Model","Discourse","T_ACC","Pronoun_ACC","ProperName_ACC","Log-lhood") )
		print("="*100)
		# Complete model
		for salience in salience_measures:
			spk = SpeakerModel( _data=formated_data,
								_vocab=vocab,
								_pronoun_type_ids=pro_type_ids,
								_lex_counts=pro_type_counts,
								_alpha=alpha,
								_decay=decay,
								_salience=salience)
			spk.predict()
			spk.evaluate(verbose=False)
			print("%12s | %15s || %2.2f | %11.2f | %14.2f | %9.2f |" % 
					("complete", salience, spk._total_acc*100, spk._pro_acc*100, spk._np_acc*100, spk._model_loglikelihood) )
		print("-"*100)
		
		# No discourse model
		spk = SM_NoDiscourse(	_data=formated_data,
								_vocab=vocab,
								_pronoun_type_ids=pro_type_ids,
								_lex_counts=pro_type_counts,
								_alpha=alpha,
								_decay=decay,
								_salience=salience)
		spk.predict()
		spk.evaluate(verbose=False)
		print("%12s | %15s || %2.2f | %11.2f | %14.2f | %9.2f |" % 
				("-discourse", "NA", spk._total_acc*100, spk._pro_acc*100, spk._np_acc*100, spk._model_loglikelihood) )
		print("-"*100)

		# No cost model
		for salience in salience_measures:
			spk = SM_NoCost( _data=formated_data,
							_vocab=vocab,
							_pronoun_type_ids=pro_type_ids,
							_lex_counts=pro_type_counts,
							_alpha=alpha,
							_decay=decay,
							_salience=salience)
			spk.predict()
			spk.evaluate(verbose=False)
			print("%12s | %15s || %2.2f | %11.2f | %14.2f | %9.2f |" % 
					("-cost", salience, spk._total_acc*100, spk._pro_acc*100, spk._np_acc*100, spk._model_loglikelihood) )
		print("-"*100)

		# No estimates of unseen referents
		for salience in salience_measures:
			spk = SM_NoUnseen(  _data=formated_data,
								_vocab=vocab,
								_pronoun_type_ids=pro_type_ids,
								_lex_counts=pro_type_counts,
								_alpha=alpha,
								_decay=decay,
								_salience=salience)
			spk.predict()
			spk.evaluate(verbose=False)
			print("%12s | %15s || %2.2f | %11.2f | %14.2f | %9.2f |" % 
					("-unseen", salience, spk._total_acc*100, spk._pro_acc*100, spk._np_acc*100, spk._model_loglikelihood) )
		print("-"*100)
	else:
		print("Finding optimum values of alpha and decay...")
		print("-------------------------------------------------------")
		alphas = [1e-4, 1e-3, 1e-2, 0.1,1,10]
		decays = [1e-2,0.1, 1, 3, 5, 10,100,1e3]
		salience_measures = ['recency','frequency']

		a_opt, d_opt = 0,0
		acc = 0

		for salience in salience_measures:
			for alpha in alphas:
				for decay in decays:
					spk = SpeakerModel(	_data=formated_data,
										_vocab=vocab,
										_pronoun_type_ids=pro_type_ids,
										_lex_counts=pro_type_counts,
										_alpha=alpha,
										_decay=decay,
										_salience=salience)
					spk.predict()
					total_acc = spk.evaluate(verbose=False)
					if total_acc>acc:
						acc = total_acc
						a_opt=alpha
						d_opt=decay
				#END-FOR-DECAY
			#END-FOR-ALPHA
			print("%s: Optimum parameters:------" % salience.upper())
			print("   alpha:",a_opt)
			print("   decay:",d_opt)
			print("Training with optimum parameters...")
			spk = SpeakerModel(	_data=formated_data,
								_vocab=vocab,
								_pronoun_type_ids=pro_type_ids,
								_lex_counts=pro_type_counts,
								_alpha=a_opt,
								_decay=d_opt,
								_salience=salience)
			spk.predict()
			spk.evaluate()
			print("-------------------------------------------------------")
		#END-FOR-SALIENCE

		## C_w = length(w)
		# RECENCY
		#  alpha*: 1e-4
		#  decay*: 100
		#  ACC: 73.90
		#	  PRP: 86.85
		#	  NNP: 70.40
		# model LLK: 3996.56
		# FREQUENCY
		#  alpha*: 1e-4
		#  ACC: 73.13
		#	  PRP: 86.85
		#	  NNP: 69.83
		# model LLK: 576.41


		## C_w = log(length(w))
		# RECENCY
		#  alpha*: 1e-4
		#  decay*: 100
		#  ACC: 75.83
		#	  PRP: 86.85
		#	  NNP: 73.18
		# model LLK: 7303.38
		# FREQUENCY
		#  alpha*: 1e-4
		#  ACC: 73.36
		#	  PRP: 79.28
		#	  NNP: 71.93
		# model LLK: 3883.20


		## C_w = log(length(w)) + 1
		# RECENCY
		#  alpha*: 1
		#  decay*: 1000
		#  ACC: 78.15
		#	  PRP: 31.08
		#	  NNP: 89.46
		# model LLK: -2304.19
		# FREQUENCY
		#  alpha*: 1
		#  ACC: 77.37
		#	  PRP: 59.36
		#	  NNP: 81.70
		# model LLK: -3968.19
