import math
import numpy as np
from abc import ABC, abstractmethod

from create_matrix import *
from formula_parser import *
from trees import Node

class Valuation(ABC):
	"""An abstract class for defining logical connectives.

	"""
	def __init__(self, variables=None):
		"""Constructor of the class `Valuation`.

		Args:
			variables (dict of str:obj): A dictionary with name:value mappings for variables.
		
		"""
		self.variables = variables
		self.values = None  # values of the last evaluation
		self.weights = None # weights of the last evaluation
		super().__init__()
	
	def evaluate(self, term):
		"""Evaluates a term.
			Calculates values and weights for all languages.

		Args:
			term (Term): A term.
		
		"""
		self.values = None
		self.weights = None
		term = self.eval([term])
		term = term[0]
		if isinstance(term, tuple):
			self.values = term[1]
			self.weights = term[0]
		else:
			try:
				self.values = term
				self.weights = np.ones(len(term))
			except TypeError:
				pass
	
	def eval(self, terms):
		"""Evaluation of a list of terms.

		Args:
			terms (list of Term): A list of term objects.
		
		Returns:
			list of float: A list of values.
		
		"""
		output = []
		for term in terms:
			if term.operator is None:
				output.append(self._var(term.operands))
			elif term.operator == "NEG":
				output.append(self._neg(self.eval(term.operands)))
			elif term.operator == "AND":
				output.append(self._and(self.eval(term.operands)))
			elif term.operator == "OR":
				output.append(self._or(self.eval(term.operands)))
			elif term.operator == "IMP":
				output.append(self._imp(self.eval(term.operands)))
			elif term.operator == "EQV":
				output.append(self._eqv(self.eval(term.operands)))
			elif term.operator == "ADD":
				output.append(self._add(self.eval(term.operands)))
		return output
	
	def collapse(self):
		"""Return the weighted average.
		"""
		nonan = np.logical_not(np.isnan(self.values))
		input = (self.weights[nonan], self.values[nonan])
		if np.sum(input[0]) == 0.0:
			return np.nan
		return np.average(input[1], weights=input[0])
	
	def collapse_std(self):
		"""Return the weighted standard deviation.
		"""
		nonan = np.logical_not(np.isnan(self.values))
		input = (self.weights[nonan], self.values[nonan])
		if np.sum(input[0]) == 0.0:
			return np.nan
		average = np.average(input[1], weights=input[0])
		return math.sqrt(np.average((input[1]-average)**2, weights=input[0]))
	
	def ones(self):
		"""Return a vector of ones (representing True).
		"""
		properties = self.variables.keys()
		n = 0
		if len(properties) > 0:
			n = len(self.variables[next(iter(properties))])
		return np.ones(n)
	
	@abstractmethod
	def _var(self, inputs):
		"""Variable
		"""
		return None
	
	@abstractmethod
	def _neg(self, inputs):
		"""Negation
		"""
		return None
	
	@abstractmethod
	def _and(self, inputs):
		"""Conjunction
		"""
		return None

	@abstractmethod
	def _or(self, inputs):
		"""Disjunction
		"""
		return None
	
	@abstractmethod
	def _imp(self, inputs):
		"""Implication
		"""
		return None

	@abstractmethod
	def _eqv(self, inputs):
		"""Equivalence
		"""
		return None

	@abstractmethod
	def _add(self, inputs):
		"""Addition
		"""
		return None


class VBoolean(Valuation):
	"""Boolean logic
	"""
	def collapse(self):
		return float(self.values)
	
	def _var(self, inputs):
		if inputs[0] == "T":
			return True
		elif inputs[0] == "F":
			return False
	
	def _neg(self, inputs):
		return not inputs[0]
	
	def _and(self, inputs):
		val = True
		for input in inputs:
			val = val and input
		return val

	def _or(self, inputs):
		val = False
		for input in inputs:
			val = val or input
		return val
	
	def _imp(self, inputs):
		return (not inputs[0]) or inputs[1]

	def _eqv(self, inputs):
		return self._imp(inputs) and self._imp(list(reversed(inputs)))

	def _add(self, inputs):
		return (inputs[0] and not inputs[1]) or (inputs[1] and not inputs[0])


class VProduct(Valuation):
	"""Product logic
	"""
	def _var(self, inputs):
		return self.variables[inputs[0]]
	
	def _neg(self, inputs):
		return 1-inputs[0]
	
	def _and(self, inputs):
		prod = np.ones(len(inputs[0]))
		for input in inputs:
			prod = np.multiply(prod, input)
		return prod

	def _or(self, inputs):
		summ = np.zeros(len(inputs[0]))
		for input in inputs:
			summ = summ+input-np.multiply(summ, input)
		return summ
	
	def _imp(self, inputs):
		return self._or([self._neg([inputs[0]]), inputs[1]])

	def _eqv(self, inputs):
		return self._and([self._imp(inputs), self._imp(list(reversed(inputs)))])

	def _add(self, inputs):
		return np.sum(inputs, axis=0)


class VProductWeighted(Valuation):
	"""Product logic with antecedent weighting
	"""
	def _var(self, inputs):
		return (self.ones(), self.variables[inputs[0]])
	
	def _neg(self, inputs):
		return (inputs[0][0], 1-inputs[0][1])
	
	def _and(self, inputs):
		prod0 = np.ones(len(inputs[0][0]))
		prod1 = np.ones(len(inputs[0][1]))
		for input in inputs:
			prod0 = np.multiply(prod0, input[0])
			prod1 = np.multiply(prod1, input[1])
		return (prod0, prod1)

	def _or(self, inputs):
		summ0 = np.zeros(len(inputs[0][0]))
		summ1 = np.zeros(len(inputs[0][1]))
		for input in inputs:
			summ0 = summ0+input[0]-np.multiply(summ0, input[0])
			summ1 = summ1+input[1]-np.multiply(summ1, input[1])
		return (summ0, summ1)
	
	def _imp(self, inputs):
		imp = self._or([self._neg([inputs[0]]), inputs[1]])
		return (inputs[0][0]*inputs[0][1]*inputs[1][0], imp[1])

	def _eqv(self, inputs):
		return self._and([self._imp(inputs), self._imp(list(reversed(inputs)))])

	def _add(self, inputs):
		return (self.ones(), np.sum([input[1] for input in inputs], axis=0))


class VFuzzy(Valuation):
	"""Fuzzy logic
	"""
	def _var(self, inputs):
		return self.variables[inputs[0]]
	
	def _neg(self, inputs):
		return 1-inputs[0]
	
	def _and(self, inputs):
		return np.min(inputs, axis=0)

	def _or(self, inputs):
		return np.max(inputs, axis=0)
	
	def _imp(self, inputs):
		return self._or([self._neg([inputs[0]]), inputs[1]])

	def _eqv(self, inputs):
		return self._and([self._imp(inputs), self._imp(list(reversed(inputs)))])

	def _add(self, inputs):
		return np.sum(inputs, axis=0)


class VFuzzyWeighted(Valuation):
	"""Fuzzy logic with antecedent weighting
	"""
	def prepare(self, term):
		return term
	
	def _var(self, inputs):
		return (self.ones(), self.variables[inputs[0]])
	
	def _neg(self, inputs):
		return (inputs[0][0], 1-inputs[0][1])
	
	def _and(self, inputs):
		args = np.argmin(inputs, axis=0)[1]
		return (np.array([inputs[x][0][i] for i, x in enumerate(args)]), np.array([inputs[x][1][i] for i, x in enumerate(args)]))

	def _or(self, inputs):
		args = np.argmax(inputs, axis=0)[1]
		return (np.array([inputs[x][0][i] for i, x in enumerate(args)]), np.array([inputs[x][1][i] for i, x in enumerate(args)]))
	
	def _imp(self, inputs):
		imp = self._or([self._neg([inputs[0]]), inputs[1]])
		return (inputs[0][0]*inputs[0][1]*inputs[1][0], imp[1])

	def _eqv(self, inputs):
		return self._and([self._imp(inputs), self._imp(list(reversed(inputs)))])

	def _add(self, inputs):
		return (self.ones(), np.sum([input[1] for input in inputs], axis=0))


def get_phylogenetic_weight(node):
	"""Calculate the weight at a phylogenetic node.

	Args:
		node (`Node`): The phylogenetic node.
	
	Returns:
		float: (weight of the parent) / (number of siblings + 1).
	
	"""
	if node.parent is None:
		return 1.0
	return 1.0*get_phylogenetic_weight(node.parent)/len(node.parent.children)


def get_phylogenetic_weights(languages):
	"""Returns the phylogenetic weights for each language.

	Args:
		languages (list of str): The list of languages.
			The languages need to have a full phylogenetic name, e.g. "IE, Germanic, German".
	
	Returns:
		list of float: The phylogenetic weights in the same order as the input languages.
	
	"""
	nodes = {"ALL" : Node("ALL", None, [])}
	for language in languages:
		language = language.split(", ")
		for i, part in enumerate(language):
			parent = "ALL"
			if i > 0:
				parent = language[i-1]
			if not part in nodes.keys():
				nodes[part] = Node(part, None, [], nodes[parent])
				nodes[parent].children.append(nodes[part])
	weights = [get_phylogenetic_weight(nodes[language.split(", ")[-1]]) for language in languages]
	return weights


def load_property_vectors(name="UD"):
	"""Load the language-property matrix and return the property vectors.

	Returns:
		dict of str:np.array: A dictionary which maps property names to property vectors.
		list of str: The list of properties.
		list of str: The list of languages in the same order as in the property vectors.
	
	"""
	matrix, languages, properties = load_language_vectors("matrices/matrix"+name+".pickle", name=name, save_overwrite=False, combine_treebanks=True)
	property_vectors = {properties[i] : v for i, v in enumerate(np.transpose(np.array(matrix, dtype=np.float64)))}
	return property_vectors, properties, languages


if __name__ == "__main__":
	property_vectors, properties, languages = load_property_vectors()
	
	formula = "( T & T ) ⇒ ( F )"
	term = parse_formula(formula)
	
	vn = VBoolean()
	vn.evaluate(term)
	print(vn.collapse())

	formula = "( D#nsubj-obj-_ & S#_-nmod ) ⇒ ( S#_-amod )"
	term = parse_formula(formula)
	
	vn = VProduct(property_vectors)
	vn.evaluate(term)
	print(vn.collapse())

	vn = VProductWeighted(property_vectors)
	vn.evaluate(term)
	print(vn.collapse())

	vn = VFuzzy(property_vectors)
	vn.evaluate(term)
	print(vn.collapse())

	vn = VFuzzyWeighted(property_vectors)
	vn.evaluate(term)
	print(vn.collapse())
