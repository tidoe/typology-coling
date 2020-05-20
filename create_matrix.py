import csv
import iso639
import lang2vec.lang2vec as l2v
import numpy as np
import os
import pandas as pd
import pickle
import pyconll
import random
import tqdm


def get_language_families(lang_file="files/language_families.txt"):
	"""Returns a dictionary with a phylogenetic path for every language.
		For example: "English" -> "IE, Germanic"
	
	Args:
		lang_file (str): The text file listing the UD languages.

	Returns:
		dict of str:str: Mappings from language names to their phylogenetic path.
	"""
	language_families = {}
	with open(lang_file, "r") as infile:
		for line in infile:
			line = line.strip()
			language = []
			for part in line.split():
				if part.isnumeric():
					break
				language.append(part)
			language = "_".join(language)
			family = line.split("K ")[1]
			language_families[language] = family
	return language_families


def full_language_name(language_families, treebank_name, combine_treebanks):
	"""Adds the phylogenetic path to a language/treebank name.
	
	Args:
		language_families (dict of str:str): Mappings from language names to their phylogenetic path.
		treebank_name (str): Name of the treebank file.
		combine_treebanks (boolean): False: the name of the treebank is used; True: the name of the language is used.
	
	Returns:
		str: The extended language/treebank name.

	"""
	language = treebank_name[3:].split("-")[0]
	family = language_families[language]
	if not combine_treebanks:
		language = treebank_name[3:]
	language = (family + ", " + language).replace("_", " ")
	return language


def load_language_vectors(filename="matrix.pickle", name="UD", save_overwrite=False, combine_treebanks=True, treebank_path="ud/ud-treebanks-v2.5/", features_sets=["syntax_wals", "fam", "geo"]):
	"""Load the language-property matrix.

	Args:
		filename (str): The name of the pickle file to load from / save to.
		name (str): "UD", "URIEL", "SP" or "ASJP"
		save_overwrite (Boolean): False: the matrix is loaded from the file; True: the matrix is calculated and the file is overwritten.
		combine_treebanks (boolean): True iff seperate treebanks for one language should be merged.
		treebank_path (str): The directory containing the treebank folders.
		features_sets (list of str): The feature sets for URIEL to load.
	
	Returns:
		list of (list of float): The language-property matrix. Languages are indices, properties are columns.
		list of str: The list of languages (= row names).
		list of str: The list of properties (= column names).

	"""
	if save_overwrite or not os.path.exists(filename):
		if name == "UD":
			matrix, languages, properties = calc_language_vectors(combine_treebanks, treebank_path)
		elif name == "URIEL":
			matrix, languages, properties = calc_language_vectors_URIEL(features_sets)
		elif name == "SP":
			matrix, languages, properties = calc_language_vectors_SP()
		elif name == "ASJP":
			matrix, languages, properties = calc_language_vectors_ASJP()
		else:
			raise ValueError("unknown name")
	else:
		with open(filename, "rb") as pf:
			matrix, languages, properties = pickle.load(pf)
	if save_overwrite:
		with open(filename, "wb") as pf:
			pickle.dump((matrix, languages, properties), pf)
	return matrix, languages, properties


def extract_counts_single(sentence, pos=False):
	"""Extract the single-edge properties from a treebank sentence.

	Args:
		`Sentence`: A sentence object.
		pos (boolean): True iff the properties should be PoS-specific.
	
	Returns:
		dict of str:(list of int): Explanation by example:
			"S#det" : [2, 0] -> An edge with the label "det" branched to the left two times.
	"""
	counts = {}
	for token in sentence:
		if token.deprel is None:
			continue
		dep_label = token.deprel.split(":")[0]
		if dep_label != "_":
			if pos:
				dep_label += "+" + token.upos
			dep_label = "S#" + dep_label
			if pos:
				pos0 = "ROOT"
				if token.head != "0":
					pos0 = sentence[token.head].upos
				dep_label = "SPOS" + dep_label[1:] + "_" + pos0
			if dep_label not in counts.keys():
				counts[dep_label] = [0, 0]
			if int(token.id) < int(token.head):
				counts[dep_label][0] += 1
			else:
				counts[dep_label][1] += 1
	return counts


def extract_counts_double(sentence, pos=False):
	"""Extract the two-edge properties with the same head from a treebank sentence.

	Args:
		`Sentence`: A sentence object.
		pos (boolean): True iff the properties should be PoS-specific.
	
	Returns:
		dict of str:(list of int): Explanation by example:
			"D#case-det" : [0, 0, 2, -> "Rows" indicate the order of the edge labels; "columns" indicate the position of the head.
			                0, 1, 0] -> Two times case-det-HEAD; one time det-HEAD-case.
							The list is only of length 3 iff the edge labels are identical.
	"""
	heads = {}
	for token in sentence:
		if token.deprel is None:
			continue
		dep_label = token.deprel.split(":")[0]
		if dep_label != "_":
			if token.head not in heads.keys():
				heads[token.head] = []
			if pos:
				dep_label += "+" + token.upos
			heads[token.head].append((token.id, dep_label))
	counts = {}
	for head in heads.keys():
		if len(heads[head]) >= 2:
			for i, dep1 in enumerate(heads[head]):
				for j, dep2 in enumerate(heads[head]):
					if int(dep1[0]) < int(dep2[0]):
						concat = "-".join(sorted([dep1[1], dep2[1]]))
						order = 0
						if concat != "-".join([dep1[1], dep2[1]]):
							order += 1
						hpos = 0
						if int(dep1[0]) < int(head):
							hpos += 1
						if int(dep2[0]) < int(head):
							hpos += 1
						label = "D#" + concat
						if pos:
							pos0 = "ROOT"
							if head != "0":
								pos0 = sentence[head].upos
							label = "DPOS" + label[1:] + "_" + pos0
						if label not in counts.keys():
							if dep1[1] == dep2[1]:
								counts[label] = [0, 0, 0]
							else:
								counts[label] = [0, 0, 0, 0, 0, 0]
						counts[label][order*3+hpos] += 1
	return counts


def extract_counts_chains(sentence, pos=False):
	"""Extract the two-edge properties with chained heads from a treebank sentence.

	Args:
		`Sentence`: A sentence object.
		pos (boolean): True iff the properties should be PoS-specific.
	
	Returns:
		dict of str:(list of int): Explanation by example:
			"C#case-det=" : [0, 0, 2, -> "Rows" indicate the order of the edge labels; "columns" indicate the position of the first head; "=" marks the second head.
			                 0, 1, 0] -> Two times case-det#2-HEAD#1; one time det#2-HEAD#1-case.
							The list is only of length 3 iff the edge labels are identical.
	"""
	heads = {}
	for token in sentence:
		if token.deprel is None:
			continue
		dep_label = token.deprel.split(":")[0]
		if dep_label != "_":
			if token.head not in heads.keys():
				heads[token.head] = []
			if pos:
				dep_label += "+" + token.upos
			heads[token.head].append((token.id, dep_label))
	counts = {}
	for head in heads.keys():
		for i, dep1 in enumerate(heads[head]):
			if dep1[0] in heads.keys():
				dep1 = (dep1[0], dep1[1] + "=")
				for j, dep2 in enumerate(heads[dep1[0]]):
					swapped = False
					if int(dep1[0]) > int(dep2[0]):
						deptmp = dep1
						dep1 = dep2
						dep2 = deptmp
						swapped = True
					concat = "-".join(sorted([dep1[1], dep2[1]]))
					order = 0
					if concat != "-".join([dep1[1], dep2[1]]):
						order += 1
					hpos = 0
					if int(dep1[0]) < int(head):
						hpos += 1
					if int(dep2[0]) < int(head):
						hpos += 1
					label = "C#" + concat
					if pos:
						pos0 = "ROOT"
						if head != "0":
							pos0 = sentence[head].upos
						label = "CPOS" + label[1:] + "_" + pos0
					if label not in counts.keys():
						if dep1[1] == dep2[1]:
							counts[label] = [0, 0, 0]
						else:
							counts[label] = [0, 0, 0, 0, 0, 0]
					counts[label][order*3+hpos] += 1
					if swapped:
						deptmp = dep1
						dep1 = dep2
						dep2 = deptmp
	return counts


def property_name(p, n=None):
	"""Returns a unique, human-readable property name.
		For example: "S#det" -> "S#_-det"; "D#case-det", 4 -> D#det-_-case (the underscore marks the position of the head).

	Args:
		p (str): The property name.
		n (int): The index.
	
	Returns:
		str: The expanded property name.
	"""
	s = p.split("#")
	d = s[0]
	s = s[1].split("_")
	p = s[0]
	pos = ""
	if len(s) > 1:
		pos += "+" + s[1]
	if d.startswith("S"):
		p = "_"+pos+"-"+p
	elif d[0] in ["D", "C"]:
		p = p.split("-")
		if n >= 3:
			p = [p[1], p[0]]
		if n % 3 == 0:
			p = "_"+pos+"-"+p[0]+"-"+p[1]
		elif n % 3 == 1:
			p = p[0]+"-_"+pos+"-"+p[1]
		elif n % 3 == 2:
			p = p[0]+"-"+p[1]+"-_"+pos
		p = p.replace("=", "_")
	else:
		p = p+"-"+str(n)
	return d+"#"+p
		

def calc_language_vectors(combine_treebanks, treebank_path):
	"""Calculate the language-property matrix.

	Args:
		combine_treebanks (boolean): True iff seperate treebanks for one language should be merged.
		treebank_path (str): The directory containing the treebank folders.
	
	Returns:
		list of (list of float): The language-property matrix. Languages are indices, properties are columns.
		list of str: The list of languages (= row names).
		list of str: The list of properties (= column names).

	"""
	counts = {}
	properties = {}
	arity = {}
	language_families = get_language_families()
	for treebank in tqdm.tqdm(os.scandir(treebank_path), total=len(os.listdir(treebank_path))):
		language = full_language_name(language_families, treebank.name, combine_treebanks)
		if language not in counts.keys():
			counts[language] = {}
		for file in os.scandir(treebank):
			if file.name.endswith(".conllu"):
				corpus = pyconll.load_from_file(file)
				for sentence in corpus:
					counts_single = extract_counts_single(sentence)
					counts_double = extract_counts_double(sentence)
					counts_chains = extract_counts_chains(sentence)
					counts_sinpos = extract_counts_single(sentence, pos=True)
					counts_doupos = extract_counts_double(sentence, pos=True)
					counts_chapos = extract_counts_chains(sentence, pos=True)
					counts_all = [counts_single, counts_double, counts_chains, counts_sinpos, counts_doupos, counts_chapos]
					counts_all = dict(sum([list(c.items()) for c in counts_all], []))	
					for p in counts_all.keys():
						if p not in counts[language].keys():
							counts[language][p] = [0 for i in range(len(counts_all[p]))]
						counts[language][p] = [counts[language][p][i]+counts_all[p][i] for i in range(len(counts_all[p]))]
						if p not in properties.keys():
							properties[p] = 0
							arity[p] = len(counts_all[p])
						properties[p] += 1
	languages = sorted(counts.keys())
	properties = sorted(properties.keys(), key=lambda k:properties[k], reverse=True)
	matrix = [[] for language in languages]
	for i, language in enumerate(languages):
		ps = []
		for j, p in enumerate(properties):
			a = arity[p]
			if a == 2:
				ps.append(property_name(p))
				matrix[i].append(None)
				if p in counts[language].keys():
					p = counts[language][p]
					matrix[i][-1] = 1.0*p[1]/(p[0]+p[1])
			else:
				for n in range(a):
					ps.append(property_name(p, n))
					matrix[i].append(None)
				if p in counts[language].keys():
					p = counts[language][p]
					for n in range(len(p)):
						matrix[i][-(a-n)] = 1.0*p[n]/sum(p)
	properties = ps
	return matrix, languages, properties


def get_language_vectors_URIEL(features_set, language_families):
	"""Returns the URIEL language vectors for a given feature set.

	Args:
		features_set (str): The name of the feature set.
		language_families (dict of str:str): Mappings from language names to their phylogenetic path.
	
	Returns:
		list of (list of float): The language-property matrix. Languages are indices, properties are columns.
		list of str: The list of languages (= row names).
		list of str: The list of properties (= column names).
	
	"""
	exceptions = {
		'aii' : 'Assyrian', 
		'arb' : 'Arabic', 
		'bho' : 'Bhojpuri', 
		'chu' : 'Old Church Slavonic', 
		'ell' : 'Greek', 
		'fro' : 'Old French', 
		'gla' : 'Scottish Gaelic', 
		'grc' : 'Ancient Greek', 
		'gsw' : 'Swiss German', 
		'gun' : 'Mbya Guarani', 
		'hsb' : 'Upper Sorbian',
		'kmr' : 'Kurmanji', 
		'koi' : 'Komi Permyak', 
		'kpv' : 'Komi Zyrian', 
		'pcm' : 'Naija', 
		'pes' : 'Persian', 
		'sme' : 'North Sami', 
		'uig' : 'Uyghur', 
		'yue' : 'Cantonese'
		# ??? : 'Buryat', 
		# ??? : 'Classical Chinese', 
		# ??? : 'Livvi', 
		# ??? : 'Hindi English', 
		# ??? : 'Old Russian',
		# ??? : 'Skolt Sami', 
		# ??? : 'Swedish Sign Language', 
	}
	vectors = {}
	for code in tqdm.tqdm(l2v.LANGUAGES):
		try:
			language = iso639.languages.get(part3=code).name
			language = language_families[language.replace(" ", "_")] + ", " + language
		except KeyError:
			if code in exceptions.keys():
				language = exceptions[code]
				language = language_families[language.replace(" ", "_")] + ", " + language
			else:
				continue
		values = l2v.get_features(code, features_set, header=False)[code]
		vectors[language] = [(None if v == "--" else v) for v in values]
	properties = l2v.get_features("eng", features_set, header=True)["CODE"]
	languages = sorted(vectors.keys())
	matrix = [vectors[language] for language in languages]
	return matrix, languages, properties


def calc_language_vectors_URIEL(features_sets):
	"""Returns the combined URIEL language vectors for given feature sets.

	Args:
		features_set (list of str): The name of the feature sets.
	
	Returns:
		list of (list of float): The language-property matrix. Languages are indices, properties are columns.
		list of str: The list of languages (= row names).
		list of str: The list of properties (= column names).
	
	"""
	language_families = get_language_families()
	matrices = []
	properties = []
	for features_set in features_sets:
		m, languages, p = get_language_vectors_URIEL(features_set, language_families)
		matrices.append(m)
		properties.extend(p)
	matrix = [[] for language in languages]
	for i in range(len(matrix)):
		for m in matrices:
			matrix[i] += m[i]
	return matrix, languages, properties


def calc_language_vectors_SP():
	"""Loads the language vectors from Serva & Petroni (2008).

	Returns:
		list of (list of str): The language-property matrix. Languages are indices, properties are columns.
		list of str: The list of languages (= row names).
		list of str: The list of properties (= column names).
	
	"""
	with open("files/serva_petroni_lists.csv") as csvfile:
		table = csv.reader(csvfile, delimiter=",", quotechar='"')
		language_families = get_language_families()
		vectors = []
		languages = []
		for i, row in enumerate(table):
			if i == 0:
				for j, cell in enumerate(row):
					vectors.append([])
					try:
						languages.append(language_families[cell] + ", " + cell)
					except KeyError:
						languages.append("")
			else:
				for j, cell in enumerate(row):
					if str(cell) == "0":
						cell = None
					vectors[j].append(cell)
		s_languages = sorted([language for language in languages if language != ""])
		matrix = []
		for language in s_languages:
			matrix.append(vectors[languages.index(language)])
		properties = [str(i+1) for i in range(len(matrix[0]))]
		return matrix, s_languages, properties


def calc_language_vectors_ASJP(path="asjp/wordlists/"):
	"""Loads the language vectors from the ASJP files.

	Args:
		path (str): The directory with the ASJP word lists.

	Returns:
		list of (list of str): The language-property matrix. Languages are indices, properties are columns.
		list of str: The list of languages (= row names).
		list of str: The list of properties (= column names).
	
	"""
	indices = [1, 2, 3, 11, 12, 18, 19, 21, 22, 23, 25, 28, 30, 31, 34, 39, 40, 41, 43, 44, 47, 48, 51, 53, 54, 57, 58, 61, 66, 72, 74, 75, 77, 82, 85, 86, 92, 95, 96, 100]
	vectors = {}
	language_families = get_language_families()
	for wordlist in tqdm.tqdm(os.scandir(path), total=len(os.listdir(path))):
		if not wordlist.name.endswith(".txt"):
			continue
		language = wordlist.name[:-4]
		language = "".join([(c if i == 0 or language[i-1] == "_" else c.lower()) for i, c in enumerate(language)])
		language = (language_families[language] + ", " + language).replace("_", " ")
		vectors[language] = [None for i in indices]
		with open(wordlist, "r") as wlist:
			for i, line in enumerate(wlist):
				if i < 2:
					continue
				line = line.split(" //")[0].split("\t")
				num = int(line[0].split(" ")[0])
				word = line[1].split(", ")[0]
				if num in indices and len(word) > 0:
					if word[0] == "%":
						word = word[1:]
					vectors[language][indices.index(num)] = word
	languages = sorted(vectors.keys())
	matrix = [vectors[language] for language in languages]
	properties = [str(i) for i in indices]
	return matrix, languages, properties


def select_language_vectors(matrix_tuple, suffixes=[""], prefixes=[""], none_replacement=None, ignore_infrequent_relations=False):
	"""Returns a submatrix of a given matrix with only the desired languages and properties.

	Args:
		matrix_tuple ((list of (list of float/str), list of str, list of str)): A tuple containing the language-property matrix, the list of languages and the list of properties.
		suffixes (list of str): A list of desired languages. Suffixes of the phylogenetic path work as well.
		prefixes (list of str): A list of desired property categories (i.e. prefixes of the property names).
		none_replacement (obj): A value to replace None values with.
		ignore_infrequent_relations (boolean): True iff 10 infrequent relations, i.e. all properties involving them, should be ignored.
	
	Returns:
		list of (list of float): The language-property matrix. Languages are indices, properties are columns.
		list of str: The list of languages (= row names).
		list of str: The list of properties (= column names).
	"""
	infrequent_relations = ["dep", "discourse", "dislocated", "fixed", "goeswith", "list", "orphan", "parataxis", "reparandum", "vocative"]
	matrix, languages, properties = matrix_tuple[0], matrix_tuple[1], matrix_tuple[2]
	matrix2 = []
	languages2 = []
	for i, language in enumerate(languages):
		for suffix in suffixes:
			if language == suffix or language.endswith(", " + suffix):
				languages2.append(language)
				matrix2.append([])
				properties2 = []
				for j, value in enumerate(matrix[i]):
					for prefix in prefixes:
						if properties[j].startswith(prefix):
							if ignore_infrequent_relations and len([r for r in infrequent_relations if r in properties[j]]) > 0:
								continue
							properties2.append(properties[j])
							if value is None:
								value = none_replacement
							matrix2[-1].append(value)
							break
				break
	return matrix2, languages2, properties2


def matrix_to_html(filename="matrix.pickle", prefixes=[""]):
	"""Creates an HTML representation of a language-property matrix.
		The output is saved to an HTML file with the same name and the selected properties ("matrix.html" in the default case).
		None values are displayed as 0.5.

	Args:
		filename (str): The name of the file with the matrix.
		prefixes (list of str): A list of desired property categories (i.e. prefixes of the property names).

	"""
	matrix, languages, properties = load_language_vectors(filename, save_overwrite=False)
	matrix, languages, properties = select_language_vectors((matrix, languages, properties), suffixes=languages, prefixes=prefixes)
	cell = None
	while cell is None:
		row = random.choice(matrix)
		cell = random.choice(row)
	if isinstance(cell, float) or isinstance(cell, int):
		df = pd.DataFrame(np.array([[(round(x, 2) if isinstance(x, float) else (0.5 if x is None else x)) for x in vec] for vec in matrix]), columns=properties, index=languages)
		html = df.style.background_gradient(cmap='coolwarm', axis=None).set_table_styles([dict(selector="th.col_heading", props=[('writing-mode', 'sideways-lr'),('min-width', '50px')])]).render()
	else:
		df = pd.DataFrame(np.array(matrix), columns=properties, index=languages)
		html = df.style.render()
	s = "+".join(sorted(list(set(prefixes))))
	s = s.replace("#", "").replace("_", "")
	if len(s) > 0:
		s = "_"+s
	with open(".".join(filename.split(".")[:-1])+s+".html", "w") as outfile:
		outfile.write(html)
