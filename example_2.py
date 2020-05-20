import numpy as np
import os

from create_matrix import *
from formula_parser import parse_formula
from search import get_phylogenetic_weights
from valuation import VFuzzy, VFuzzyWeighted


if os.path.exists("matrices/matrixUD.pickle"):
	# If precomputed language vectors exists, load them:
	matrix, languages, properties = load_language_vectors("matrices/matrixUD.pickle", name="UD", combine_treebanks=True, save_overwrite=False)

else:
	# Otherwise use a toy matrix:
	matrix = [
		# SOV  Prep
		[0.00, 0.04], # IE, Germanic, English
		[0.46, 0.01], # IE, Germanic, German
		[0.09, 0.00], # IE, Romance, French
		[0.07, 0.00], # IE, Romance, Italian
		[0.16, 0.01], # IE, Romance, Spanish
		[0.96, 1.00]  # Japanese, Japanese
	]
	languages = ["English", "German", "French", "Italian", "Spanish", "Japanese"]
	properties = ["D#nsubj-obj-_", "S#_-case"]
	
	# Get mappings from languages to their phylogenetic path:
	language_families = get_language_families()
	
	# Update language names, e.g. "English" -> "IE, Germanic, English"
	languages = [language_families[language] + ", " + language for language in languages]


# Get property vectors from language vectors:

property_vectors = {properties[i] : v for i, v in enumerate(np.transpose(np.array(matrix, dtype=np.float64)))}


# Define a formula (spaces around operators and brackets are important):

formula = "D#nsubj-obj-_ ⇒ ( ¬ ( ¬ S#_-case ) )"
term = parse_formula(formula)


# Instantiate a valuation for fuzzy logic:

valuation = VFuzzy(property_vectors)


# Evaluate the formula:

valuation.evaluate(term)

# The valuation saves the values and weights for every language in valuation.values and valuation.weights, respectively.
# In `VFuzzy` all weights are set to 1 (unweighted average) after each evaluation.

# Therefore, the `collapse` function calculates the unweighted average:

print(valuation.collapse(), valuation.collapse_std())


# For phylogenetic weighting, calculate the phylogenetic weight for each language
# (note that the language names must be full phylogenetic paths, e.g. "IE, Germanic, English"):
pweights = np.array(get_phylogenetic_weights(languages))


# Update `valuation.weights` and calculate the average:

valuation.weights = np.multiply(valuation.weights, pweights)
print(valuation.collapse(), valuation.collapse_std())
