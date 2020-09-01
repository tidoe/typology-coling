import os
import pyconll
import shutil
from sklearn import svm, tree

# Define the changes that should be made:
# language to change : ([treebanks to change towards UD format], [treebanks already close to UD format], [relations to change])
changes = {
	"Chinese" : (["GSD", "GSDSimp", "PUD"], ["CFL", "HK"], ["clf"]),
	"Korean" : (["GSD"], ["Kaist", "PUD"], ["aux"]),
	"Galician" : (["CTG"], ["TreeGal"], ["nummod", "cc"])
}

# treebank version (automatically extended to "ud/" + treebank_path)
treebank_path = "ud-treebanks-v2.5"

def main():
	"""Main method.
		Delete all previously changed treebanks to avoid mixing. Then changes the treebanks as specified.
		If more than one relations should be changed, treebanks for all combinations of changed relations are created.
		The input treebanks are searched in "ud/"; the output treebanks are saved in "ud-harmony/".

	"""
	shutil.rmtree("ud-harmony/" + treebank_path)
	os.mkdir("ud-harmony/" + treebank_path)
	for language in list(changes.keys()):
		change = changes[language]
		for i in range(1, 1 << len(change[2])):
			relations = [change[2][j] for j in range(len(change[2])) if (i & (1 << j))]
			for t in change[0]:
				indir = "ud/" + treebank_path + "/UD_" + language + "-" + t
				outdir = "ud-harmony/" + treebank_path + "/UD_" + language + "-" + t
				if not os.path.exists(outdir):
					os.mkdir(outdir)
				outdir += "/" + "+".join(relations)
				if not os.path.exists(outdir):
					os.mkdir(outdir)
				for file in os.scandir(indir):
					if file.name.endswith(".conllu"):
						corpus = pyconll.load_from_file(file)
						corpus = change_corpus(language, corpus, relations)
						with open(outdir + "/" + file.name, "w") as f:
							f.write(corpus.conll())

def change_corpus(language, corpus, relations):
	"""Perform changes in a corpus.
		Container method to load language-specific methods.
	
	Args:
		language (str): The language to change.
		corpus (`Corpus`): The treebank (Pyconll corpus object) to change.
		relations (list of str): The relations to change.
	
	Returns:
		`Corpus`: The changed treebank.
	
	"""
	if language == "Chinese":
		if "clf" in relations:
			corpus = change_chinese_clf(corpus)
	elif language == "Korean":
		if "aux" in relations:
			corpus = change_korean_aux(corpus)
	elif language == "Galician":
		if "nummod" in relations:
			corpus = change_galician_nummod(corpus)
		if "cc" in relations:
			corpus = change_galician_cc(corpus)
	return corpus

def change_chinese_clf(corpus):
	"""Changes instances of the "clf" relation towards the UD annotations scheme.
		Only apply this method to: "GSD", "GSDSimp", "PUD"!
		
		For each instance of "clf":
			Let its dependent be the classifier.
			Let its head be the noun.
			Let the dependents of the classifier be the modifiers (mostly only a numeral or a determiner).
			If there is exactly one modifier:
				Change the head of the classifier to the modifier.
				Change the head of the modifier to the noun.
			Elif there are no modifiers:
				Change the classifier relation (from "clf") to "det".
			# There is a very small number of instances with more than one modifiers, but these are not handled.
	
	Args:
		corpus (`Corpus`): The treebank to change.
	
	Returns:
		`Corpus`: The changed treebank.

	"""
	for sentence in corpus:
		for token in sentence:
			if token.deprel is None:
				continue
			dep_label = token.deprel
			if dep_label == "clf":
				classifier = token.id
				noun = token.head
				modifier = []
				for token2 in sentence:
					if token2.head == classifier:
						modifier.append((token2.id, token2.deprel))
				if len(modifier) == 1:
					modifier = modifier[0]
					sentence[classifier].head = modifier[0]
					sentence[modifier[0]].head = noun
				elif len(modifier) == 0:
					sentence[classifier].deprel = "det"
	return corpus

def change_korean_aux(corpus):
	"""Changes instances of the "aux" relation towards the UD annotations scheme.
		Only apply this method to: "GSD"!
		
		For each instance of "flat":
			Let its dependent be the auxiliary.
			Let its head be the main verb.
			If the UPOS tags for both auxiliary and main verb are "VERB":
				Change the "flat" relation to "aux".
				Change the UPOS tag of the auxiliary to "AUX".
	
	Args:
		corpus (`Corpus`): The treebank to change.
	
	Returns:
		`Corpus`: The changed treebank.

	"""
	for sentence in corpus:
		for token in sentence:
			if token.deprel is None:
				continue
			dep_label = token.deprel
			if dep_label == "flat":
				verb = sentence[token.head]
				aux = token
				if verb.upos == "VERB" and aux.upos == "VERB":
					sentence[aux.id].deprel = "aux"
					sentence[aux.id].upos = "AUX"
	return corpus

def change_galician_cc(corpus):
	"""Changes instances of the "cc" relation towards the UD annotations scheme.
		Only apply this method to: "CTG"!
		
		For each conjunction construction (a conjunction construction contains at least one "cc" instance):
			Let the head of all "cc" relations be the first conjunct.
			For all non-"cc" dependents of the first conjunct:
				Decide whether it is a conjunct (and not a modifier, for example)
				# The decision is made by a decision tree trained on the conjunction constructions of the TreeGal train set.
			Change the labels of the conjuncts to "conj".
			Change the head of the "cc" dependents (from the first conjunct) to the directly following conjunct (if there is one).
	
	Args:
		corpus (`Corpus`): The treebank to change.
	
	Returns:
		`Corpus`: The changed treebank.

	"""
	X, y = _change_galician_cc_extract_feature_vectors("train")
	clf = tree.DecisionTreeClassifier(random_state=1)
	clf.fit(X, y)
	#X, y = _change_galician_cc_extract_feature_vectors("test")
	#print(clf.score(X, y)) # accuracy: 0.932
	for sentence in corpus:
		conjunctions = {}
		for token in sentence:
			if token.deprel is None:
				continue
			dep_label = token.deprel
			if dep_label == "cc":
				if token.head not in conjunctions.keys():
					conjunctions[token.head] = []
				conjunctions[token.head].append(token.id)
		for conjunct1 in conjunctions.keys():
			ccs = conjunctions[conjunct1]
			conjuncts = []
			for token2 in sentence:
				if token2.head == conjunct1 and token2.deprel != "cc":
					pre_token = _change_galician_cc_token_before_subtree(sentence, token2)
					if clf.predict([_change_galician_cc_mask(sentence[conjunct1].upos, token2.upos, int(token2.id)-int(conjunct1), pre_token in ccs)])[0] == 1:
						conjuncts.append((token2.id, token2.deprel, token2.upos, pre_token))
			for conjunct in conjuncts:
				sentence[conjunct[0]].deprel = "conj"
				if sentence[conjunct[3]].deprel == "cc":
					sentence[conjunct[3]].head = conjunct[0]
	return corpus
	
def change_galician_nummod(corpus):
	"""Changes instances of the "nummod" relation towards the UD annotations scheme.
		Only apply this method to: "CTG"!
		
		For each instance of "nummod":
			Let its dependent be the numeral.
			Let its head be the determiner.
			Let the head of the determiner be the noun.
			Change the head of the numeral (from the determiner) to the noun.
	
	Args:
		corpus (`Corpus`): The treebank to change.
	
	Returns:
		`Corpus`: The changed treebank.

	"""
	for sentence in corpus:
		for token in sentence:
			if token.deprel is None:
				continue
			dep_label = token.deprel
			if dep_label == "nummod":
				numeral = token.id
				determiner = token.head
				noun = sentence[determiner].head
				sentence[numeral].head = noun
	return corpus

pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

def _change_galician_cc_mask(pos1, pos2, dist, preceding_cc):
	"""Create a feature vector.
		POS tags are converted to one-hot vectors.
	
	Args:
		pos1 (str): POS of the head
		pos2 (str): POS of the dependent
		dist (int): directed distance between head and dependent
		preceding_cc (boolean): True iff dependent is directly preceded by a "cc" dependent belonging to the conjunction construction
	
	Returns:
		list of int: Feature vector.
	
	"""
	d1 = [0 for tag in pos_tags]
	d1[pos_tags.index(pos1)] = 1
	d2 = [0 for tag in pos_tags]
	d2[pos_tags.index(pos2)] = 1
	return d1 + d2 + [dist, (1 if preceding_cc else 0)]

def _change_galician_cc_extract_feature_vectors(file):
	"""Extract feature vectors for the non-"cc" dependents of the first conjunct in a conjunction.
		(The dependents can be conjuncts of the conjunction or other dependents of the first conjunct.)
		
	Args:
		file (str): "train" or "test"
			Load the according TreeGal treebank.
	
	Returns:
		list of (list of int): The feature vectors.
		list of int: The classes.
			1 for conjuncts.
			0 for other dependents.

	"""
	true_conjuncts = []
	false_conjuncts = []
	train_corpus = pyconll.load_from_file("ud/" + treebank_path + "/UD_Galician-TreeGal/gl_treegal-ud-" + file + ".conllu")
	for sentence in train_corpus:
		for token in sentence:
			if token.deprel is None:
				continue
			dep_label = token.deprel
			if dep_label == "cc":
				conjunction = token.id
				conjunct_ = token.head
				conjunct1 = sentence[conjunct_].head
				if conjunct1 != "0":
					for token2 in sentence:
						if token2.head == conjunct1:
							tokens = [token2.id]
							ccs = []
							a = True
							while a:
								a = False
								for t in sentence:
									if t.head in tokens and t.id not in tokens + ccs:
										if t.deprel == "cc":
											ccs.append(t.id)
										else:
											tokens.append(t.id)
										a = True
							tokens = sorted(tokens)
							d = _change_galician_cc_mask(sentence[conjunct1].upos, token2.upos, int(token2.id)-int(conjunct1), str(int(tokens[0])-1) in ccs)
							if token2.deprel == "conj":
								true_conjuncts.append(d)
							else:
								false_conjuncts.append(d)
	X = true_conjuncts + false_conjuncts
	y = [1 for x in true_conjuncts] + [0 for x in false_conjuncts]
	return X, y

def _change_galician_cc_token_before_subtree(sentence, token):
	"""Determine the token directly preceding a subtree in a sentence.
	
	Args:
		sentence (`Sentence`): The sentence.
		token (`Token`): The root token of the subtree.
	
	Returns:
		str: The ID of the token directly preceding the root token and all of its dependents.

	"""
	tokens = [token.id]
	a = True
	while a:
		a = False
		for t in sentence:
			if t.head in tokens and t.id not in tokens:
				tokens.append(t.id)
				a = True
	tokens = sorted(tokens)
	return str(int(tokens[0])-1)

if __name__ == "__main__":
	main()