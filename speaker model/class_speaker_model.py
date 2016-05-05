import numpy as np
import scipy
from utils import  third_person_prp, pronoun_type_debug, pronounBydeprel, pronounBytype
from sklearn.metrics import classification_report, accuracy_score
import ipdb
import pickle
import os,sys
sys.path.append('../')
sys.path.append('../topicality\ and\ salience/')
from hdp.ugs import *
#from utils_topicality import 


class SpeakerModel:
	def __init__(self,_data,
					_vocab,
					_pronoun_type_ids,
					_lex_counts,
					_topicmodel,
					_alpha=0.1,_decay=3, _salience='recency'):
		# corpus  Ndocs x entities
		self.data = _data
		self._D = len(_data)
		# vocab: list of phrases, used to map phr_id in corpus
		self.vocab = _vocab # np array
		self.vocab_size = len(_vocab)
		# number of referents
		self._n_referents = self.vocab_size - len(third_person_prp)
		# pronoun type (e.g. singular_male) ids in same order as vocabulary
		self.pronoun_type_ids = _pronoun_type_ids
		# number of pronoun types
		self._n_pro_type = len(pronoun_type_debug) - 1 # don't count NOT FOUND
		# counts of unseen referents by pronoun types, got from lexicon
		self._u_pro_type = np.zeros(self._n_pro_type)
		# total number of unseen referents
		self._u_total = 0

		# parameter for prob of new referent
		self._alpha = _alpha
		# decay for discourse salience
		self._decay = _decay
		# discourse salience measure, p(r)| vals: ['freq','rec']
		self._salience = _salience
		# topic model for topicality measure
		self._topicmodel = _topicmodel
		
		# number of pronouns and proper names to predict
		self._n_samples = sum([len(doc) for doc in self.data])
		# V: number of ref. expressions that can refer to r, constant across referents
		self._V = self._n_samples / self._n_referents
		# word likelihood given referent p(w|r)
		self._p_w_r = 1.0 / self._V

		#class labels
		self.class_labels = ['PRP','NNP']
		# true class
		self._Y 	 = []
		# predicted class
		self._Y_pred = []

		#calc_unseen_counts lex_counts
		self.calc_unseen_counts(_lex_counts)

		# save metrics
		self._model_loglikelihood = 0.0
		self._total_acc = 0.0
		self._np_acc = 0.0		# proper name accuracy
		self._pro_acc = 0.0		# pronoun accuracy


	def calc_unseen_counts(self,_lex_counts):
		self._u_pro_type = _lex_counts[:self._n_pro_type]
		self._u_total = _lex_counts[-1]

	def speech_cost(self,word_id):
		#return len( self.vocab[word_id] )
		return np.log( len(self.vocab[word_id]) )
		#return np.log( len(self.vocab[word_id]) )+1.0

	
	'''
	p(r) : discourse salience of referent r up until now
	'''
	def get_salience(self,referent_id,counts_state,last_mention_dist,doc_id=-1):
		p_r = 0
		ref_prev_mentions = counts_state[referent_id] if len(counts_state)!=0 else 0
		if ref_prev_mentions==0:			# new referent
			pro_type = self.pronoun_type_ids[referent_id]
			p_r = self._alpha * self._u_pro_type[pro_type] / self._u_total
		else:
			if self._salience=='frequency':
				p_r = ref_prev_mentions						# frequency measure
			elif self._salience=='recency':					# 'recency'
				p_r = np.exp(-last_mention_dist/self._decay)	# recency measure
			elif self._salience=='topicality':
				# "Document" == current referent counts state --> not considered
				word_id = self._topicmodel._vocab.index(referent_id)
				logprob_xk = self._topicmodel.assign_topic_word(word_id) # log p(x|t,k)
				# now calculate prob of all topics on document D
				logprob_kd = self._topicmodel.topic_proportion_by_doc(doc_id) # theta_doc
				# pick most probable topic assignment: p(x|k)*p(k|d)
				logprob_x = logprob_xk + logprob_kd
				k = logprob_x.argmax()
				#topicality: prob of topic within document
				p_r = np.exp(logprob_kd[k])
		return p_r

	'''
	Sum over all potencial referents compatible with w. Sum_{r'} {p(w|r')*p(r')}
	'''
	def get_sum_potencial_referents(self,pos,referent_id,counts_state,mention_state,global_pos,doc_id):
		# proper noun only refers to itself
		if pos[0]=='N':
			last_mention_dist = global_pos - mention_state[referent_id]
			return self.get_salience(referent_id,counts_state,last_mention_dist,doc_id)
		# pronoun spotted
		else:
			pro_type = self.pronoun_type_ids[referent_id]
			# get potencial referents ids in vocabulary
			cond = self.pronoun_type_ids==pro_type
			potencial_refs_ids = []
			for i,val in enumerate(cond):
				if val and self.vocab[i] not in third_person_prp:
					potencial_refs_ids.append(i)

			# sum over all potencial referents
			sum_p_r = 0
			for ref_id in potencial_refs_ids:
				last_mention_dist = global_pos - mention_state[ref_id]
				prev_mentions = counts_state[ref_id] if len(counts_state)!=0 else 0
				if prev_mentions!=0:												# sum only active referents
					sum_p_r +=  self.get_salience(ref_id,counts_state,last_mention_dist,doc_id)

			if sum_p_r==0:		# if PRP and there is no referent mentioned before
				return 0.0		# return 
			sum_p_r += self.get_salience(referent_id,[],0,doc_id)		# add unseen entity prob

			return sum_p_r

	"""
	Predict between pronoun or proper name for each referent as discourse advances.
	"""
	def predict(self):
		self._model_loglikelihood = 0.0		# calculate model log likelihood as it predicts
		for doc_id in range(self._D):
			doc = self.data[doc_id]
			referent_counts = np.zeros(1000)
			last_mention = np.zeros(1000)
			for entity in doc:
				dep_rel = entity.dep_rel
				ref_id = entity.referent_id
				pro_type_name = pronoun_type_debug[ self.pronoun_type_ids[ref_id] ]

				true_pos = entity.pos[:3]
				# true label
				self._Y.append(self.class_labels.index(true_pos))
				# predicting refering expression (POS)
				pred_pos = self.class_labels.index('NNP')	# pick proper name by default

				#print("ref_exp:",self.vocab[entity.phrase])
				#print("referent:",self.vocab[ref_id])

				# discourse salience for referent r
				last_mention_distance = entity.global_pos - last_mention[ref_id]
				p_r = self.get_salience(referent_id=ref_id,
										counts_state=referent_counts,
										last_mention_dist=last_mention_distance,
										doc_id=doc_id)

				# PROPER NAME CASE
				sum_p_r_np = self.get_sum_potencial_referents('NNP', 
										ref_id,
										referent_counts,
										last_mention,
										entity.global_pos,
										doc_id)
				cw_np = self.speech_cost(ref_id)
				log_speaker_np =  -np.log(sum_p_r_np) - np.log(cw_np)

				if abs(log_speaker_np)!=np.inf:
					self._model_loglikelihood += log_speaker_np

				#ipdb.set_trace()

				# PRONOUN CASE
				sum_p_r_pro = self.get_sum_potencial_referents('PRP',
										ref_id,
										referent_counts,
										last_mention,
										entity.global_pos,
										doc_id)
				# speaker's cost: cross agreement and grammatical position
				log_speaker_pro = -np.inf
				for pro in pronounBydeprel[dep_rel]:
					if pro in pronounBytype[pro_type_name]:
						# valid pronouns
						pro_id = np.nonzero(self.vocab==pro)[0][0]
						cw_pro = self.speech_cost(pro_id)

						local_log_speaker_pro = 0.0
						if sum_p_r_pro==0:		# no referent active
							local_log_speaker_pro = -np.inf
						else:
							local_log_speaker_pro = -np.log(sum_p_r_pro) - np.log(cw_pro)
						log_speaker_pro = max(log_speaker_pro,local_log_speaker_pro)

						# add model log likelihood only if Ps != 0
						if local_log_speaker_pro!=-np.inf:
							self._model_loglikelihood += local_log_speaker_pro

				if log_speaker_pro > log_speaker_np:
					pred_pos = self.class_labels.index('PRP')

				# predicted label
				self._Y_pred.append(pred_pos)

				#update counts
				referent_counts[ref_id] += 1
				last_mention[ref_id] = entity.global_pos

				#print("Predicted label:",self.class_labels[pred_pos])
				#ipdb.set_trace()
			#END-FOR-DOC
		#END-FOR-DOCUMENTS
		self._Y = np.array(self._Y)
		self._Y_pred = np.array(self._Y_pred)


	def evaluate(self,verbose=True):
		class_accs = []
		for i in range(len(self.class_labels)):
			label = self.class_labels[i]
			acc_class = sum((self._Y==i) * (self._Y_pred==i)) / sum(self._Y==i)
			if label[0]=='P':
				self._pro_acc = acc_class
			else:
				self._np_acc  = acc_class
			class_accs.append(acc_class)
			if verbose:
				print("Accuracy %s: %.2f" % (label,acc_class*100))
		self._total_acc = accuracy_score(self._Y,self._Y_pred)
		if verbose:
			print("Total accuracy: %.2f" % (self._total_acc*100) )
			print("Model log likelihood: %.2f" % self._model_loglikelihood)
		if 0 in class_accs:
			return 0
		return self._total_acc

