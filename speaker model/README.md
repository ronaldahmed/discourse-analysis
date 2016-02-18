<b>Discourse influence in speakers' choice of referring expressions</b>

<p>Based on the paper:</p>

<p>
Naho Orita, Eliana Vornov, Naomi Feldman, Hal Daumé III. (2015). Why discourse affects speakers' choice of referring expressions. Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics.
</p>

<p>The paper proposes a language production model to incorporate update to listeners's belief as discourse proceeds, incorporating discourse salience, cost of speech production and probabilities of unseen referents (obtained from an external lexicon).</p>
<p>The dataset used was the training set for the <a href="http://stel.ub.edu/semeval2010-coref/node/7">first task</a> of the <a href="http://stel.ub.edu/semeval2010-coref/">SemEval 2010</a>. In addition, the named entity list of <a href="http://www.clsp.jhu.edu/~sbergsma/Gender/">Bergsma and Lin (2006)</a> is used as lexicon of pronoun types (singular-masculine, singular-feminine, singular-neutral, and plural). </p>

<p>The model parameters (alpha and decay) are tuned running several configurations over the corpus. The tuned parameter values differ from those stated in the paper due to lack of extra data (CHILDES and extra agreement annotation for OntoNotes), manually annotated by the authors.</p>

<p>Files</p>
<ul>
	<li><b>main_speaker_model.py:</b><br/>
		Reads the data and run the models described in N. Orita et all (2015).
	</li>
	<li><b>utils.py:</b><br/>
		Functions to read and preprocess the data and resources.
	</li>
	<li><b>class_speaker_model.py:</b><br/>
		Complete model, incorporating discourse salience, speech cost, and unseen referents probabilities.
	</li>
	<li><b>alternative_models.py:</b><br/>
		Alternative language production models for comparison. These models are the following: model without salience (uniform salience instead of recency or frequency salience); model without speech cost (constant cost); and model without good estimates of unseen referents (uniform estimate for all pronoun types).
	</li>
	<li><b>pro_type_counts.pickle</b></br>
		Counts by pronoun type (singular-masculine, singular-feminine, singular-neutral, and plural), obtained from <a href="http://www.clsp.jhu.edu/~sbergsma/Gender/">Bergsma and Lin (2006)</a>. Necessary to calculate the unseen entities probabilities.
	</li>
	<li><b>names_in_gnlexicon.pickle</b></br>
		Intersection of vocabularies from <a href="http://www.clsp.jhu.edu/~sbergsma/Gender/">Bergsma and Lin (2006)</a> and training corpus.
	</li>
	<li><b>pro_type_ids.pickle</b></br>
		Automatic agreement annotation (pronoun type) for all referents in the corpus.
	</li>
	<li></li>
</ul>
