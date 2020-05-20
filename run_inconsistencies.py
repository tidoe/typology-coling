import numpy as np
import os
import scipy.cluster.hierarchy
import sys

from create_matrix import *

if __name__ == "__main__":

	matrixUDT, languagesUDT, propertiesUDT = load_language_vectors("matrices/matrixUDT.pickle", name="UD", save_overwrite=(not os.path.exists("matrices/matrixUDT.pickle")), combine_treebanks=False)
	
	languages = languagesUDT
	matrixUD, languagesUD, propertiesUD = matrixUDT, languagesUDT, propertiesUDT
	
	lv_single, languages, q = select_language_vectors((matrixUD, languagesUD, propertiesUD), suffixes=languages, prefixes=["S#"], none_replacement=None)
	
	infrequent_relations = ["dep", "discourse", "dislocated", "fixed", "goeswith", "list", "orphan", "parataxis", "reparandum", "vocative"]
	print("find inconsistencies")
	properties = q
	scores = {}
	for t, treebank in enumerate(languages):
		language = treebank.split(", ")[-1].split("-")[0]
		if language not in scores.keys():
			scores[language] = {p : [] for p in properties}
		for i, p in enumerate(properties):
			scores[language][p].append(lv_single[t][i])
	res = []
	for language in scores.keys():
		for i, p in enumerate(properties):
			if len([r for r in infrequent_relations if r in p]) > 0:
				continue
			values = scores[language][p]
			values = [v for v in values if v is not None]
			if len(values) < 2:
				continue
			res.append((language, p, max(values)-min(values), [round(v, 2) for v in values]))
	res = sorted(res, key=lambda x: x[2])
	for r in res:
		print(r)
