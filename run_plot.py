import matplotlib.pyplot as plt
import numpy as np

from valuation import load_property_vectors

vecs, props, langs = load_property_vectors()

fs = 16

x = vecs["DPOS#_+VERB-nsubj+NOUN-obj+NOUN"]
y = 1-vecs["SPOS#_+NOUN-case+ADP"]
plt.scatter(x, y, marker="o", c=np.maximum(1-x, y), vmin=0, vmax=1, cmap='coolwarm')
#plt.scatter(x, y, marker="o", c=np.maximum(1-x, y), vmin=0, vmax=1, cmap='Blues')
plt.xlabel("verb-nsubj:noun-obj:noun (VSO)", fontsize=fs)
plt.ylabel("case:adp-noun (prepositions)", fontsize=fs)
plt.savefig("coling/u3.png")

plt.clf()

x = vecs["DPOS#nsubj+NOUN-obj+NOUN-_+VERB"]
y = vecs["SPOS#_+NOUN-case+ADP"]
plt.scatter(x, y, marker="o", c=np.maximum(1-x, y), vmin=0, vmax=1, cmap='coolwarm')
#plt.scatter(x, y, marker="o", c=np.maximum(1-x, y), vmin=0, vmax=1, cmap='Blues')
plt.xlabel("nsubj:noun-obj:noun-verb (SOV)", fontsize=fs)
plt.ylabel("noun-case:adp (postpositions)", fontsize=fs)
plt.savefig("coling/u4.png")
