import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

from formula_parser import *
from valuation import *
from search import get_phylogenetic_weights

universals = [
	(1, "S-O"),
	(2, "Adp-NP ⇔ N-Gen"),
	(3, "V-S-O ⇒ Adp-NP"),
	(4, "S-O-V ⇒ NP-Adp"),
	(5, "( S-O-V & N-Gen ) ⇒ N-Adj"),
	(6, "V-S-O ⇒ ( V-S-O + S-V-O )"),
	(7, "S-O-V ⇒ Adv-V"),
	(8, None),
	(9, None),
	(10, None),
	(11, None),
	(12, None),
	(13, "O-V ⇒ VP-V"),
	(14, None),
	(15, None),
	(16, "( V-S-O ⇒ Aux-V ) & ( S-O-V ⇒ V-Aux )"),
	(17, "V-S-O ⇒ N-Adj"),
	(18, "Adj-N ⇒ ( Dem-N & Num-N )"),
	(19, None),
	(20, None),
	(21, "Adj-Adv ⇒ ( N-Adj & V-O )"),
	(22, None),
	(23, None),
	(24, "Rel-N ⇒ ( NP-Adp | Adj-N )"),
	(25, "V-O_pron ⇒ V-O")
]

variables = {
	"Adj-Adv" : "SPOS#_+ADJ-advmod+ADV",
	"Adj-N" : "( ¬ SPOS#_+NOUN-amod+ADJ )",
	"Adp-NP" : "( ¬ SPOS#_+NOUN-case+ADP )",
	"Adv-V" : "( ¬ SPOS#_+VERB-advmod+ADV )",
	"Aux-V" : "( ¬ SPOS#_+VERB-aux+AUX )",
	"Dem-N" : "( ¬ SPOS#_+NOUN-det+PRON )",
	"Gen-N" : "( ¬ SPOS#_+NOUN-nmod+NOUN )",
	"N-Adj" : "SPOS#_+NOUN-amod+ADJ",
	"N-Gen" : "SPOS#_+NOUN-nmod+NOUN",
	"NP-Adp" : "SPOS#_+NOUN-case+ADP",
	"Num-N" : "( ¬ SPOS#_+NOUN-nummod+NUM )",
	"O-S-V" : "DPOS#obj+NOUN-nsubj+NOUN-_+VERB",
	"O-V" : "( ¬ SPOS#_+VERB-obj+NOUN )",
	"Rel-N" : "( ¬ SPOS#_+NOUN-acl+VERB )",
	"S-O" : "( DPOS#_+VERB-nsubj+NOUN-obj+NOUN + DPOS#nsubj+NOUN-_+VERB-obj+NOUN + DPOS#nsubj+NOUN-obj+NOUN-_+VERB )",
	"S-O-V" : "DPOS#nsubj+NOUN-obj+NOUN-_+VERB",
	"S-V-O" : "DPOS#nsubj+NOUN-_+VERB-obj+NOUN",
	"V-Aux" : "SPOS#_+VERB-aux+AUX",
	"V-O" : "SPOS#_+VERB-obj+NOUN",
	"V-O_pron" : "SPOS#_+VERB-obj+PRON",
	"V-S-O" : "DPOS#_+VERB-nsubj+NOUN-obj+NOUN",
	"VP-V" : "( ¬ SPOS#_+VERB-xcomp+VERB )"
}

def experiment1():
	"""Evaluate Greenberg's universals on the UD treebanks
	"""
	property_vectors, properties, languages = load_property_vectors()
		
	vn = VFuzzy(property_vectors)
		
	pweights = np.array(get_phylogenetic_weights(languages))

	results = []
	for i, formula in universals:
		if formula is None:
			continue
		formula = " " + formula + " "
		for v in variables.keys():
			formula = formula.replace(" " + v + " ", " " + variables[v] + " ")
		term = parse_formula(formula.strip())
		vn.evaluate(term)
		vn.weights = np.multiply(vn.weights, pweights)
		results.append((i, vn.collapse(), vn.collapse_std()))

	results = sorted(results, key=lambda x: x[1], reverse=True)
	for i, avg, std in results:
		formula = [u[1] for u in universals if u[0] == i][0]
		formula = " " + formula + " "
		for v in variables.keys():
			formula = formula.replace(" " + v + " ", " " + r"\text{"+v+r"}" + " ")
		formula = formula.strip()
		formula = formula.replace("¬", r"\lnot").replace("&", r"\land").replace("|", r"\lor").replace("⇒", r"\Rightarrow").replace("⇔", r"\Leftrightarrow")
		formula = "$" + formula + "$"
		print(" & ".join(["%.2f" % avg, "%.2f" % std, formula, r"\#"+str(i)]) + r"\\")

def build_weights(train_test_vec, pweights, train_test):
	"""Builds a weight vector.

	Args:
		train_test_vec (list of boolean): Boolean value which indicates for each language whether it belongs to the train set (true) or the test set (false)
		pweights (list of float): (phylogenetic) weights for all languages in the train/test set.
		train_test (boolean): True if the weight vector is for the train set; False if it is for the test set.
	
	Returns:
		list of float: Weights for all languages.

	"""
	pweights_ = []
	j = 0
	for t in train_test_vec:
		if t == train_test:
			pweights_.append(pweights[j])
			j += 1
		else:
			pweights_.append(0)
	return pweights_

def select_random_languages(languages, p, level="language"):
	"""Split languages into a train and a test set.

	Args:
		languages (list of str): The list of languages.
			The languages need to have a full phylogenetic name, e.g. "IE, Germanic, German".
		p (float): The relative size of the train set.
		level (str): The split level (language/subfamily/family).
	
	Returns:
		nparray: Weight vector for the train set (languages of the test set have a weight of 0).
		nparray: Weight vector for the test set (languages of the train set have a weight of 0).
	
	"""
	if level == "language":
		n = int(round(p*len(languages)))
		train_test_vec = [True] * n + [False] * (len(languages)-n)
		random.shuffle(train_test_vec)
	else:
		if level == "subfamily":
			groups = [language.split(", ")[-2] for language in languages]
		elif level == "family":
			groups = [language.split(", ")[0] for language in languages]
		else:
			raise ValueError("Unknown level: " + level)
		groups_ = sorted(list(set(groups)))
		n = int(round(p*len(groups_)))
		group_vec = [True] * n + [False] * (len(groups_)-n)
		random.shuffle(group_vec)
		train_test_vec = [group_vec[groups_.index(groups[i])] for i, language in enumerate(languages)]
	pweights = get_phylogenetic_weights([language for i, language in enumerate(languages) if train_test_vec[i]])
	pweights_train = build_weights(train_test_vec, pweights, True)
	pweights = get_phylogenetic_weights([language for i, language in enumerate(languages) if not train_test_vec[i]])
	pweights_test = build_weights(train_test_vec, pweights, False)
	return np.array(pweights_train), np.array(pweights_test)

def run_split_evaluation(formulas, vn, languages, p, level):
	"""Run a random-split evaluation.

	Args:
		formulas (list of str): List of formulas.
		vn (Valuation): Valuation.
		languages (list of str): The list of languages.
			The languages need to have a full phylogenetic name, e.g. "IE, Germanic, German".
		p (float): The relative size of the train set.
		level (str): The split level (language/subfamily/family).
	
	Returns:
		float: Root-Mean-Square Error
		float: Root-Mean-Square Error (s_1>=90% condition)
		float: Percentage of values that have been removed because they were NaN.

	"""
	train_avg, train_std, test_avg, test_std = [], [], [], []
	nans = 0
	for formula in formulas:
		formula = " " + formula + " "
		for v in variables.keys():
			formula = formula.replace(" " + v + " ", " " + variables[v] + " ")
		term = parse_formula(formula.strip())
		pweights_train, pweights_test = select_random_languages(languages, p, level)
		vn.evaluate(term)
		vn.weights = np.multiply(vn.weights, pweights_train)
		avg1 = vn.collapse()
		std1 = vn.collapse_std()
		vn.evaluate(term)
		vn.weights = np.multiply(vn.weights, pweights_test)
		avg2 = vn.collapse()
		std2 = vn.collapse_std()
		if not (np.isnan(avg1) or np.isnan(avg2)):
			train_avg.append(avg1)
			train_std.append(std1)
			test_avg.append(avg2)
			test_std.append(std2)
		else:
			nans += 1
	nans = 1.0*nans/len(formulas)
	
	train_avg = np.array(train_avg)
	test_avg = np.array(test_avg)
	rmse = np.sqrt(np.mean(np.square(train_avg-test_avg), axis=None))
	
	train_avg_90 = train_avg[train_avg >= 0.9]
	test_avg_90 = test_avg[train_avg >= 0.9]
	rmse_90 = np.sqrt(np.mean(np.square(train_avg_90-test_avg_90), axis=None))
	
	plt.plot(train_avg, test_avg, ",")
	x = np.linspace(0, 1, 1000)
	plt.plot(x, 1*x+0)
	plt.savefig("coling/" + level[0].upper() + str(int(100*p)) + ".png")
	plt.clf()
	
	return rmse, rmse_90, nans

def experiment2():
	"""Run six random-split experiments.

	"""
	formula_containers = set()

	for i, formula in universals:
		if formula is None:
			continue
		for variable in sorted(variables.keys(), key=lambda x: len(x), reverse=True):
			formula = formula.replace(variable, "*")
		if not "+" in formula:
			formula_containers.add(formula)
	
	formulas = []
	for formula in formula_containers:
		formula_length = len(formula)-len(formula.replace("*", ""))
		var_lists = itertools.permutations(variables.keys(), formula_length)
		for var_list in var_lists:
			formulax = formula
			for v in var_list:
				formulax = formulax.replace("*", v, 1)
			formulas.append(formulax)
	print(len(formulas))

	property_vectors, properties, languages = load_property_vectors()
		
	vn = VFuzzy(property_vectors)

	print(run_split_evaluation(formulas, vn, languages, 0.5, "language"))
	print(run_split_evaluation(formulas, vn, languages, 0.2, "language"))
	print(run_split_evaluation(formulas, vn, languages, 0.5, "subfamily"))
	print(run_split_evaluation(formulas, vn, languages, 0.2, "subfamily"))
	print(run_split_evaluation(formulas, vn, languages, 0.5, "family"))
	print(run_split_evaluation(formulas, vn, languages, 0.2, "family"))
	"""
	204226
	(0.1267627801082895, 0.07160239878920738, 0.0)
	(0.14939479966692853, 0.11571128372222127, 0.0)
	(0.1361986356408813, 0.07952699197608773, 0.0)
	(0.16116775828124527, 0.13282686584147907, 0.0)
	(0.17107725946425628, 0.11231114973153034, 0.0)
	(0.21952409129202516, 0.2292076094538831, 0.001958614476119593)
	"""
	
if __name__ == "__main__":
	args = sys.argv
	if len(args) > 1:
		mode = args[1]
		if mode == "1":
			experiment1()
		elif mode == "2":
			experiment2()
	else:
		print("Command line argument required! You have 2 options:")
		print("1: Evaluate Greenberg's universals on the UD treebanks")
		print("2: Run six random-split experiments")