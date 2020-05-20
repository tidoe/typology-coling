import itertools
import numpy as np
from multiprocessing import Pool

from create_matrix import *
from formula_parser import *
from valuation import *


def load_property_vectors(ud_prefixes, uriel_prefixes):
	"""Load the language-property matrix and return the property vectors.

	Args:
		ud_prefixes (list of str): A list of desired UD property categories (i.e. prefixes of the property names).
		uriel_prefixes (list of str): A list of desired URIEL property categories (i.e. prefixes of the property names).
	
	Returns:
		dict of str:np.array: A dictionary which maps property names to property vectors.
		list of str: The list of properties in alphabetically sorted order.
		list of str: The list of languages in the same order as in the property vectors.
	
	"""
	# load matrices
	matrixUD, languagesUD, propertiesUD = load_language_vectors("matrices/matrixUD.pickle", name="UD", save_overwrite=False)
	matrixURIEL, languagesURIEL, propertiesURIEL = load_language_vectors("matrices/matrixURIEL.pickle", name="URIEL", save_overwrite=False)
		
	# select languages
	languages = sorted(list(set(languagesUD).intersection(set(languagesURIEL))))
		
	# select properties
	matrixUD, languagesUD, propertiesUD = select_language_vectors((matrixUD, languagesUD, propertiesUD), suffixes=languages, prefixes=ud_prefixes, none_replacement=None)
	matrixURIEL, languagesURIEL, propertiesURIEL = select_language_vectors((matrixURIEL, languagesURIEL, propertiesURIEL), suffixes=languages, prefixes=uriel_prefixes, none_replacement=None)
		
	property_vectors = {}

	# refine UD property selection
	untinteresting_relations = []#["dep", "discourse", "dislocated", "fixed", "goeswith", "list", "orphan", "parataxis", "reparandum", "vocative"] + ["compound", "flat", "punct", "root"]
	c = {"S" : 100, "D" : 600, "C" : 600}
	matrixUD = np.transpose(np.array(matrixUD, dtype=np.float64))
	for i, v in enumerate(matrixUD):
		p = propertiesUD[i]
		if c[p[0]] > 0 and len([r for r in untinteresting_relations if r in p]) == 0:
			property_vectors[p] = v
			c[p[0]] -= 1
			if p[0] in ["D", "C"] and (i+1) % 6 == 0:
				px = p.split("#")
				ps = px[1].split("-")
				hd, d1, d2 = ps[2], ps[1], ps[0]
				property_vectors[px[0]+"#"+hd+"-ini;["+d1+","+d2+"]"] = matrixUD[i-2]+matrixUD[i-5]
				property_vectors[px[0]+"#"+hd+"-med;["+d1+","+d2+"]"] = matrixUD[i-1]+matrixUD[i-4]
				property_vectors[px[0]+"#"+hd+"-fin;["+d1+","+d2+"]"] = matrixUD[i-0]+matrixUD[i-3]
				if p[0] == "D":
					property_vectors[px[0]+"#"+d1+"-"+d2+";["+hd+"]"] = matrixUD[i-3]+matrixUD[i-4]+matrixUD[i-5]
					# property_vectors[px[0]+"#"+d2+"-"+d1+";["+hd+"]"] = matrixUD[i-0]+matrixUD[i-1]+matrixUD[i-2]

	# refine URIEL property selection
	matrixURIEL = np.transpose(np.array(matrixURIEL, dtype=np.float64))
	for i, v in enumerate(matrixURIEL):
		p = propertiesURIEL[i]
		if p[0] == "S" or (p[0] == "F" and sum(v) > 1):
			property_vectors[p] = v

	return property_vectors, sorted(property_vectors.keys()), languages


def build_network(variables, max_vars):
	"""Generates disjunctive normal forms (DNFs) by iteratively expanding nodes, 
		starting with variables, applying negation, conjunction and disjunction to nodes.
		Don't use because very slow and runtime-inefficient!

	Args:
		variables (list of str): A list of variable names.
		max_vars (int): The maximum number of variables involved in a formula.
			All appearances of a variable are counted individually.
	
	Returns:
		dict of str:Term: The network, i.e. a mapping from formulas in DNF to respective terms.
	
	"""
	nodes = {variable : Term([variable]) for variable in variables}
	checked = set()
	while True:
		ks = list(nodes.keys())
		ks.sort()
		new_nodes = {}
		for i, k1 in enumerate(ks):
			term1 = nodes[k1]
			if k1 not in checked:
				term_neg = convert_to_dnf(Term([term1], "NEG"))
				term_neg_str = traverse_inorder(term_neg)
				if term_neg_str not in ["0", "1"] and len(term_neg.atoms()) <= max_vars:
					new_nodes[term_neg_str] = term_neg
				checked.add(k1)
			for j, k2 in enumerate(ks):
				if j >= i:
					break
				if (k1,k2) not in checked:
					term2 = nodes[k2]
					term_con = convert_to_dnf(Term([term1, term2], "AND"))
					term_con_str = traverse_inorder(term_con)
					if term_con_str not in ["0", "1"] and len(term_con.atoms()) <= max_vars:
						new_nodes[term_con_str] = term_con
					term_dis = convert_to_dnf(Term([term1, term2], "OR"))
					term_dis_str = traverse_inorder(term_dis)
					if term_dis_str not in ["0", "1"] and len(term_dis.atoms()) <= max_vars:
						new_nodes[term_dis_str] = term_dis
					checked.add((k1,k2))
		len1 = len(nodes)
		nodes.update(new_nodes)
		len2 = len(nodes)
		if len2 == len1:
			break
	return nodes


class SearchState():
	"""A class to store information about disjunctive normal forms (DNFs), which are search states.
	
	"""
	def __init__(self, term, values, length, disjuncts):
		"""Constructor of the class `SearchState`.

		Args:
			term (Term): The term of the DNF.
			values (np.array): The values after evaluating the formula for every language.
			length (int): The length (in terms of variables) of the formula.
			disjuncts (set of (tuple of str)): The disjuncts of the formula.
				A disjunct is represented as a tuple which contains the names of its atomic variables in sorted order.
				Negated variables are appened by "NEG ".
		
		"""
		self.term = term
		self.values = values
		self.length = length
		self.disjuncts = disjuncts
	
	def __str__(self):
		"""String representation of the class `SearchState`.

		Returns:
			str: inorder traversal of the DNF, vector, length, set of disjuncts as stringified tuple
		
		"""
		return str((traverse_inorder(self.term), self.values, self.length, self.disjuncts))


vn_averaging = VProduct() # global variable
def average_value(values, weights):
	"""Calculate the weighted average of a vector.

	Args:
		values (np.array): The value vector.
		weights (np.array): The weight vector.
	
	Returns:
		float: The weighted average.
	
	"""
	vn_averaging.values = values
	vn_averaging.weights = weights
	return vn_averaging.collapse()


def worthy(values, weights, nan_p=0.25, t=0.5):
	"""Checks whether a vector is worth to be kept.

	Args:
		values (np.array): The vector.
		weights (np.array): The corresponding weights.
		nan_p (float): Maximum percentage of NaN values.
		t (float): Minimum average value.
	
	Returns:
		boolean: True iff the vector should be kept.
	
	"""
	return np.count_nonzero(np.isnan(values)) <= nan_p*len(values) and average_value(values, weights) >= t


def sort_out_ones(states, weights, max_me=0.05):
	"""Separates final states from non-final states.
		States with an average value close to 1 are final.

	Args:
		states (list of SearchState): List of states.
		weights (np.array): Weight vector for averaging.
		max_me (float): Maximum distance to 1.
	
	Returns:
		list of SearchState: List of states with an average value close to 1.
		list of SearchState: List of the other states.
	
	"""
	ones = []
	no_ones = []
	for state in states:
		if average_value(1-state.values, weights) <= max_me:
			ones.append(state)
		else:
			no_ones.append(state)
	return ones, no_ones


def improved(operator, state1, state2, values, weights, i=0.05):
	"""Checks whether the connection of two states significantly improved the average value.
		disjunction: the new average value is greater than the greatest average value of the state to connect
		conjunction: the new average value is smaller than the smallest average value of the state to connect

	Args:
		operator (str): "OR" or "AND", the connective
		state1 (SearchState): 1st state to connect
		state2 (SearchState): 2nd state to connect
		values (np.array): vector of the connection
		weights (np.array): weights for weighted averaging
		i (float): minimum improvement
	
	Returns:
		boolean: True iff the connection shows an improvement.

	"""
	m1 = average_value(state1.values, weights)
	m2 = average_value(state2.values, weights)
	m = average_value(values, weights)
	if operator == "OR":
		l = len(state1.disjuncts)+len(state2.disjuncts)
		return m >= max(m1, m2) + (l-1)*i
	elif operator == "AND":
		l = len(next(iter(state1.disjuncts)))+len(next(iter(state2.disjuncts)))
		return m <= min(m1, m2) - (l-1)*i
	return None


def minimal_disjuncts(dnf1, dnf2):
	"""Checks whether the disjunction of two DNFs has minimal disjuncts, i.e.
		1) there is no atom common to all disjuncts.
		2) there are no two disjuncts which are identical except for one atom which
			is contained as unnegated atom in one disjunct and as negated atom in the other disjunct.
	
	Args:
		dnf1 (SearchState): 1st DNF
		dnf2 (SearchState): 2nd DNF
	
	Returns:
		boolean: True iff the DNFs can be combined without causing a non-minimal DNF.

	"""
	disjuncts = dnf1.disjuncts | dnf2.disjuncts
	if len(set.intersection(*[set(c) for c in disjuncts])) > 0:
		return False
	for c1, c2 in itertools.combinations(disjuncts, 2):
		if 2*(len(set(c1+c2))-1) == len(c1+c2):
			x = list(set(c1).difference(set(c2)) | set(c2).difference(set(c1)))
			if len(x) == 2 and x[0].replace("NEG ", "") == x[1].replace("NEG ", ""):
				return False
	return True


def search(property_vectors, languages, weighted=True, max_vars=4, k_best=100, max_me=0.05, tr_th=0.5, improve=0.05):
	"""Searach disjunctive normal forms (DNFs) with maximal average scores.

	Args:
		property_vectors (dict of str:np.array): A dictionary which maps property names to property vectors.
		languages (list of str): The languages with values in the property vectors.
		weighted (boolean): True iff phylogenetic weights should be used.
		max_vars (int): The maximum number of variables involved in a formula.
			All appearances of a variable are counted individually.
		k_best (int): Only keeps the k best disjuncts when iteratively constructing DNFs.
		max_me (float): Maximum distance to 1 for final states.
		tr_th (float): Minimum average value for disjuncts.
		improve (float): Minimum improvement per disjunct.
	
	Returns:
		list of SearchState: List of ranked final states.
	
	"""
	valuation = VFuzzy(property_vectors)

	pweights = np.ones(len(languages))
	if weighted:
		pweights = np.array(get_phylogenetic_weights(languages))
	pweights = np.array(pweights)
	
	# construct atoms
	atoms = []
	for p in sorted(property_vectors.keys()):
		# type(p) can be `str` or `numpy.str_` but must be `str`
		atoms.append(SearchState(Term([str(p)]), property_vectors[p], 1, set([(str(p),)])))
		atoms.append(SearchState(Term([Term([str(p)])], "NEG"), 1-property_vectors[p], 1, set([("NEG " + str(p),)])))
	
	atoms = [atom for atom in atoms if worthy(atom.values, pweights, t=tr_th)]
	final_states, atoms = sort_out_ones(atoms, pweights, max_me=max_me)
	
	dnfs = [atoms]
	
	# construct disjuncts (conjunctions)
	for n in range(max_vars-1):
		conjuncts = dnfs[n]
		new_disjuncts = []
		for conjunct in conjuncts:
			conjunct_atoms = sorted(conjunct.term.atoms(True))
			for atom in dnfs[0]:
				concatenated_atoms = conjunct_atoms + atom.term.atoms(True)
				if atom.term.atoms()[0] not in conjunct.term.atoms() and tuple(concatenated_atoms) == tuple(sorted(concatenated_atoms)):
					operands = (conjunct.term.operands if n > 0 else [conjunct.term])
					values = valuation._and([conjunct.values, atom.values])
					if worthy(values, pweights, t=tr_th):# and improved("AND", conjunct, atom, values, pweights, i=improve):
						new_disjuncts.append(SearchState(Term(operands + [atom.term], "AND"), values, 2+n, set([tuple(sorted(concatenated_atoms))])))
		dnfs.append(new_disjuncts)
	
	dnfs = sum(dnfs, [])
	
	# construct disjunctions (DNFs)
	for n in range(max_vars-1):
		dnfs = sorted(dnfs, key=lambda dnf: average_value(dnf.values, pweights), reverse=True)
		dnfs = dnfs[:k_best]
		new_dnfs = []
		for i, dnf1 in enumerate(dnfs):
			disjuncts1 = (dnf1.term.operands if dnf1.term.operator == "OR" else [dnf1.term])
			for j, dnf2 in enumerate(dnfs):
				if j >= i:
					break
				disjuncts2 = (dnf2.term.operands if dnf2.term.operator == "OR" else [dnf2.term])
				if dnf1.length + dnf2.length <= max_vars and len(dnf1.disjuncts.intersection(dnf2.disjuncts)) == 0 and minimal_disjuncts(dnf1, dnf2):
					values = valuation._or([dnf1.values, dnf2.values])
					if worthy(values, pweights, t=tr_th) and improved("OR", dnf1, dnf2, values, pweights, i=improve):
						new_dnfs.append(SearchState(Term(disjuncts1 + disjuncts2, "OR"), values, dnf1.length + dnf2.length, dnf1.disjuncts | dnf2.disjuncts))
		new_final_states, new_dnfs = sort_out_ones(new_dnfs, pweights, max_me=max_me)
		final_states.extend(new_final_states)
		dnfs.extend(new_dnfs)
	
	final_states.extend(dnfs)
	
	final_states = sorted(final_states, key=lambda state: average_value(state.values, pweights), reverse=True)
	return final_states
