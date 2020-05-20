import numpy as np
import pandas as pd
import re
import seaborn as sns

from formula_parser import *
from valuation import *
from search import *

if __name__ == "__main__":
	property_vectors, _, languages = load_property_vectors(["S#"], [])
	valuation = VFuzzy(property_vectors)
	pweights = np.array(get_phylogenetic_weights(languages))
	atoms = {}
	for p in property_vectors.keys():
		p = str(p)
		px = p.replace("S#", "")
		if px.replace("-", "").replace("_", "") not in ["acl", "advmod", "amod", "aux", "case", "det", "nmod", "nsubj", "nummod", "obj"]:
			continue
		atoms[px] = Term([p])
		px = "-".join(list(reversed(px.split("-"))))
		atoms[px] = Term([Term([p])], "NEG")
	matrix = [[None for atom in atoms] for atom in atoms]
	a_keys = sorted(atoms.keys(), key=lambda x: x.replace("-", "").replace("_", "") + "_" * x.split("-").index("_"))
	imps = []
	for i, a1 in enumerate(a_keys):
		for j, a2 in enumerate(a_keys):
			valuation.evaluate(Term([atoms[a1], atoms[a2]], "IMP"))
			valuation.weights = pweights
			val = valuation.collapse()
			std = valuation.collapse_std()
			if np.isnan(val):
				val = 0
			else:
				val = round(val, 2)
			matrix[i][j] = val
			if i < j:
				imps.append((val, std, a1 + u" ⇒ " + a2))
	imps = sorted(imps, key=lambda x: x[0], reverse=True)
	for value, std, formula in imps[:30]:
		print(value, std, formula)
	df = pd.DataFrame(np.array(matrix), columns=a_keys, index=a_keys)
	mn = min(sum(matrix, []))
	mx = max(sum(matrix, []))
	html = df.style.background_gradient(cmap='coolwarm', axis=None, low=1.0*mn/((mx-mn)**2), high=1.0*(1-mx)/((mx-mn)**2)).set_table_styles([dict(selector="th.col_heading", props=[('writing-mode', 'sideways-lr'),('min-width', '50px')])]).render()
	#html = df.style.background_gradient(cmap='Blues', axis=None).set_table_styles([dict(selector="th.col_heading", props=[('writing-mode', 'sideways-lr'),('min-width', '50px')])]).render()
	with open("matrices/implications.html", "w") as outfile:
		outfile.write(html)
	#hm = sns.heatmap(df, annot=False)
	#hm.figure.savefig("run_implications.png")