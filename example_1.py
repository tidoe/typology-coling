import numpy as np
import os
import scipy.cluster.hierarchy

from create_matrix import *
from trees import *


# Load matrices if existing else_compute and save them:

matrixUD, languagesUD, propertiesUD = load_language_vectors("matrices/matrixUD.pickle", name="UD", combine_treebanks=True, save_overwrite=(not os.path.exists("matrices/matrixUD.pickle")))
matrixURIEL, languagesURIEL, propertiesURIEL = load_language_vectors("matrices/matrixURIEL.pickle", name="URIEL", features_sets=["syntax_wals", "fam", "geo"], save_overwrite=(not os.path.exists("matrices/matrixURIEL.pickle")))
matrixSP, languagesSP, propertiesSP = load_language_vectors("matrices/matrixSP.pickle", name="SP", save_overwrite=(not os.path.exists("matrices/matrixSP.pickle")))


# Convert the language-property matrices to HTML files (to view them in a broweser):

matrix_to_html("matrices/matrixUD.pickle", prefixes=["S#"])      # UD: single-link properties
matrix_to_html("matrices/matrixURIEL.pickle", prefixes=["F_"])   # URIEL: phylogenetic properties
matrix_to_html("matrices/matrixSP.pickle")                       # Serva and Petroni (2008)

# prefix  matrix  meaning
# S#      UD      single-link property
# D#      UD      double-link property
# C#      UD      chain-link property
# SPOS#   UD      single-link property with POS
# DPOS#   UD      double-link property with POS (shouldn't be converted to HTML because it's too large)
# CPOS#   UD      chain-link property with POS ( --- same --- )
# S_      URIEL   WALS property
# F_      URIEL   phylogenetic property
# GC_     URIEL   geographic property


# Reduce a matrix to specific languages and properties:

languages = ["English", "Swedish", "Danish", "German", "Dutch", "Romanian", "French", "Italian", "Spanish", "Portuguese", "Latvian", "Lithuanian", "Polish", "Slovak", "Czech", "Slovenian", "Bulgarian"]
lv_single, languages, _ = select_language_vectors((matrixUD, languagesUD, propertiesUD), suffixes=languages, prefixes=["S#"])
lv_family, languages, _ = select_language_vectors((matrixURIEL, languagesURIEL, propertiesURIEL), suffixes=languages, prefixes=["F_"])
lv_string, languages, _ = select_language_vectors((matrixSP, languagesSP, propertiesSP), suffixes=languages)

# Now, `lv_single` contains the UD single-link vectors, `lv_family` contains the phylogenetic vectors from URIEL and `lv_string` contains the conceptual vectors (i.e. word lists) from Serva and Petroni (2008) for the specified list of languages.


# Cluster the language vectors:

scipy.cluster.hierarchy._EUCLIDEAN_METHODS = () # allow custom Euclidean distance
Z_single = linkage(lv_single, method="ward", metric=lambda x, y: euclidean(x, y))
Z_family = linkage(lv_family, method="ward", metric=lambda x, y: euclidean(x, y))
d = list(set(sum([x for x in lv_string], []))) # strings must be encoded as integers for the linkage function, then decoded to the actual strings inside the distance metric
Z_string = linkage([[d.index(w) for w in x] for x in lv_string], method="average", metric=lambda x, y: td(x, y, d))

# trees.py also contains the function `load_linkage` which loads a precomputed linkage if existing.


# Save the dendrograms:

save_dendrogram("dendros/single.png", Z_single, languages)
save_dendrogram("dendros/family.png", Z_family, languages)
save_dendrogram("dendros/string.png", Z_string, languages)


# Compute tree distances:

# choose a gold tree
Z_gold = Z_string

# compute random tree distance
d_rand = random_tree_distance(len(languages), 1000, weighted=False)

# compute and print normalised and unnormalised tree distance for single-link vectors
d_single = tree_distance(Z_gold, Z_single, weighted=False)
print("single", "%.2f" % (1-1.0*((d_rand-d_single)/d_rand)), "%.2f" % d_single)

# compute and print normalised and unnormalised tree distance for phylogenetic vectors
d_family = tree_distance(Z_gold, Z_family, weighted=False)
print("family", "%.2f" % (1-1.0*((d_rand-d_family)/d_rand)), "%.2f" % d_family)