import os,sys
import ipdb
import matplotlib
import matplotlib.pyplot as plt
sys.path.append("../")
from utils import *

data_conll_dir = "../datasets/semeval_2010_t1_eng/data"
data_noun_lexicon_dir = "../datasets/noun_gender_number/"

if __name__=="__main__":
	print(read_noun_lexicon(data_noun_lexicon_dir))