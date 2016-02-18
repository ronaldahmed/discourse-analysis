from class_speaker_model import SpeakerModel
import numpy as np

class SM_NoDiscourse(SpeakerModel):
	'''
	p(r) : Uniform discourse salience
	'''
	def get_salience(self,referent_id,ref_prev_mentions,last_mention_dist):
		return 1.0 / self._n_referents

class SM_NoCost(SpeakerModel):
	def speech_cost(self,word_id):
		return 1

class SM_NoUnseen(SpeakerModel):
	def calc_unseen_counts(self,_lex_counts):
		self._u_pro_type = np.ones(self._n_pro_type)
		self._u_total = 1		#cte