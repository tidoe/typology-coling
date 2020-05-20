import Levenshtein as ls
import math
import numpy as np
import os
import pickle
import random
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage
from scipy.spatial import distance
from scipy.stats import zscore


# random weighted (True) and unweighted (False) tree distances for 10.000 runs
PD_RAND = {
	16 : {True :  430.3, False :  1241.2},
	17 : {True :  490.4, False :  1473.2},
	65 : {True : 8077.4, False : 48126.6}
}

# Pruned tree from Serva & Petroni (2008)
TREE_GOLD = [
	[
		[
			"English",
			[
				[
					"Swedish",
					"Danish"
				],
				[
					"German",
					"Dutch"
				]
			]
		],
		[
			"Romanian",
			[
				"French",
				[
					"Italian",
					[
						"Spanish",
						"Portuguese"
					]
				]
			]
		]
	],
	[
		[
			"Latvian",
			"Lithuanian"
		],
		[
			[
				"Polish",
				[
					"Slovak",
					"Czech"
				]
			],
			[
				"Slovenian",
				"Bulgarian"
			]
		]
	]
]

# Tree from Rabinovich et al. (2017)
TREE_RABINOVICH = [
	[
		[
			"Italian",
			[
				"French",
				"Spanish"
			]
		],
		[
			[
				"German",
				"Dutch"
			],
			[
				"English",
				[
					"Swedish",
					"Danish"
				]
			]
		]
	],
	[
		[
			"Romanian",
			"Lithuanian"
		],
		[
			[
				"Portuguese",
				[
					"Czech",
					"Slovak"
				]
			],
			[
				"Bulgarian",
				[
					"Latvian",
					[
						"Polish",
						"Slovenian"
					]
				]
			]
		]
	]
]

# Tree from Bjerva et al. (2019)
TREE_BJERVA = [
	[
		[
			[
				[
					[
						"Swedish",
						"Danish"
					],
					"English"
				],
				"Romanian"
			],
			[
				[
					"Slovenian",
					"Czech"
				],
				[
					"Slovak",
					"Polish"
				]
			]
		],
		[
			[
				#[
					"Dutch",
					"German"
				#],
				#"Greek"
			],
			[
				[
					"Portuguese",
					"Spanish"
				],
				[
					"Italian",
					"French"
				]
			]
		]
	],
	[
		#[
			#[
				#"Finnish",
				#"Estonian"
			#],
			#[
				"Latvian",
				"Lithuanian"
			#]
		#],
		#"Hungarian"
	]
]


def euclidean(x, y, remove_nans=True):
	"""Computes the Euclidean distance between two 1-D arrays.

	Args:
		x (list of float): Input vector.
		y (list of float): Input vector.
		remove_nans (boolean): True iff NaN values should be ignored.
	
	Returns:
		float: The Euclidean distance between vectors x and y.
	
	"""
	if remove_nans:
		s = 0.0
		d = 0
		for i in range(len(x)):
			if np.isnan(x[i]) or np.isnan(y[i]):
				continue
			s += (x[i]-y[i]) ** 2
			d += 1
		if d > 0:
			s = math.sqrt(s*len(x)/d)
		return s
	return distance.euclidean(x, y)


def td(x, y, d):
	"""Calculates the 'temporal distance' for two (masked) vectors of words.

	Args:
		x (list of int): The 1st vector.
		y (list of int): The 2nd vector.
		d (dict of int:str): The mapping from integer keys to words.
	
	Returns:
		float: The TD between x and y.

	"""
	x = [d[i] for i in x.astype('int64')]
	y = [d[i] for i in y.astype('int64')]
	x1 = [v for i, v in enumerate(x) if x[i] is not None and y[i] is not None]
	y1 = [v for i, v in enumerate(y) if x[i] is not None and y[i] is not None]
	return -1750*np.log(1-1.09*(1.0-ls.seqratio(x1, y1)))


def ldnd(x, y, d):
	"""Calculates the 'divided normalised Levensthein distance' for two (masked) vectors of words.

	Args:
		x (list of int): The 1st vector.
		y (list of int): The 2nd vector.
		d (dict of int:str): The mapping from integer keys to words.
	
	Returns:
		float: The LDND between x and y.

	"""
	x = [d[i] for i in x.astype('int64')]
	y = [d[i] for i in y.astype('int64')]
	x1 = [v for i, v in enumerate(x) if x[i] is not None and y[i] is not None]
	y1 = [v for i, v in enumerate(y) if x[i] is not None and y[i] is not None]
	x2 = []
	y2 = []
	for i, v in enumerate(x):
		for j, w in enumerate(y):
			if i != j and v is not None and w is not None:
				x2.append(v)
				y2.append(w)
	return (1.0-ls.seqratio(x1, y1))/(1.0-ls.seqratio(x2, y2))


def load_linkage(filename, matrix, method, metric, save_overwrite=False):
	"""Loads a linkage from a file or calculates it.

	Args:
		filename (str): The filename.
		list of (list of float): The language-property matrix.
		method (str): As in scipy's 'linkage'.
		metric (func): As in scipy's 'linkage'.
		save_overwrite (Boolean): False: the linkage is loaded from the file; True: the linkage is calculated and the file is overwritten.
	
	Returns:
		ndarray: The linkage.

	"""
	if save_overwrite or not os.path.exists(filename):
		Z = linkage(matrix, method=method, metric=metric)
	else:
		with open(filename, "rb") as pf:
			Z = pickle.load(pf)
	if save_overwrite:
		with open(filename, "wb") as pf:
				pickle.dump(Z, pf)
	return Z


def evaluate_clustering(Z, gold_clusters):
	"""Calculates precision, recall, f-score and purity between a predicted and a gold clustering.

	Args:
		Z (ndarray): The linkage for the gold clustering.
		gold_clusters (dict of int:int): The assignment from elements to clusters.
			Keys must have the same meaning as the elements in the linkage.
	
	Returns:
		float: Clustering precision.
		float: Clustering recall.
		float: Clustering f-score.
		float: Clustering purity.

	"""
	n = len(set([gold_clusters[k] for k in gold_clusters.keys()]))
	pred_clusters = list(cut_tree(Z, n_clusters=n))
	pred_clusters = {i : int(list(pred_clusters[i])[0]) for i in range(len(pred_clusters))}
	tp, tn, fp, fn = 0, 0, 0, 0
	for i in gold_clusters.keys():
		for j in gold_clusters.keys():
			if i > j:
				continue
			if gold_clusters[i] == gold_clusters[j]:
				if pred_clusters[i] == pred_clusters[j]:
					tp += 1
				else:
					fn += 1
			else:
				if pred_clusters[i] == pred_clusters[j]:
					fp += 1
				else:
					tn += 1
	p = 1.0*tp/(tp+fp)
	r = 1.0*tp/(tp+fn)
	f = 2.0*p*r/(p+r)
	cluster_sizes = {}
	for i in pred_clusters.keys():
		cr = pred_clusters[i]
		if cr not in cluster_sizes.keys():
			cluster_sizes[cr] = {}
		cl = gold_clusters[i]
		if cl not in cluster_sizes[cr].keys():
			cluster_sizes[cr][cl] = 0
		cluster_sizes[cr][cl] += 1
	purity = 0
	for cr in cluster_sizes.keys():
		purity += max(list(cluster_sizes[cr].values()))
	purity /= 1.0*len(pred_clusters)
	return p, r, f, purity


def save_dendrogram(filename, Z, labels):
	"""Saves a dendrogram to a file.

	Args:
		filename (str): The name of the file to save the dendrogram to.
		Z (ndarray): The linkage for the dendrogram.
		labels (list of str): List of leaf labels.
	
	"""
	fig = plt.figure(figsize=(20, len(labels)/2.0))
	ax = fig.add_subplot(1, 1, 1)
	dendrogram(Z, orientation='left', labels=labels, ax=ax)
	ax.tick_params(axis='x', which='major', labelsize=20)
	ax.tick_params(axis='y', which='major', labelsize=20)
	fig.subplots_adjust(left=0.05, right=0.65)
	fig.savefig(filename)


class Node():
	"""A class for a dendrogram node.

	"""
	def __init__(self, name, distance, children, parent=None):
		"""Constructor of the dendrogram node class.

		Args:
			name (int): The index of the node.
			distance (float): The distance between the node and the bottom of the dendrogram (= 0).
			children (list of `Node`): The children of the node. Leaf nodes have 0 children.
			parent (`Node`): The parent of the node.
			
		"""
		self.name = name
		self.distance = distance
		self.children = children
		self.parent = parent


def tree_distance(Z1, Z2, weighted=False, norm=zscore):
	"""Calculates the squared path-difference distance between two given dendrograms.

	Args:
		Z1 (ndarray): Linkage of the 1st dendrogram.
		Z2 (ndarray): Linkage of the 2nd dendrogram.
		weighted (boolean): True if the weighted distance should be calculated; False if all edges have a weight of 1.
		norm (func): Normalisation for the distance vectors. Not used if weighted=False.
	
	Returns:
		float: The squared path-difference distance.
	
	"""
	if not weighted:
		norm = lambda x: x
	dv1 = norm(distance_vector(Z1, weighted))
	dv2 = norm(distance_vector(Z2, weighted))
	return distance.euclidean(dv1, dv2) ** 2


def tree_distance_from_tree(Z, structure, languages):
	"""Calculates the squared path-difference distance between a given and a constant tree.

	Args:
		Z (ndarray): Linkage of the given dendrogram.
		structure (list of obj): TREE_GOLD, TREE_RABINOVICH or TREE_BJERVA
		languages (list of str): The languages in the correct oder (for indexing).
			The list must contain exactly those languages which are leaves in the structure.
	
	Returns:
		float: The squared path-difference distance.
	
	"""
	dv1 = distance_vector(Z, weighted=False)
	dv2 = distance_vector_from_tree(structure, languages)
	return distance.euclidean(dv1, dv2) ** 2


def random_tree_distance(n, m, weighted=False, norm=zscore):
	"""Calculates the average squared path-difference distance between two random dendrograms.

	Args:
		n (int): The number of leaves in the dendrograms.
		m (int): The number of runs to average over.
		weighted (boolean): True if the weighted distance should be calculated; False if all edges have a weight of 1.
		norm (func): Normalisation for the distance vectors. Not used if weighted=False.
	
	Returns:
		float: The average squared path-difference distance.
	
	"""
	if not weighted:
		norm = lambda x: x
	d = 0
	for i in range(m):
		dv1 = norm(random_distance_vector(n, weighted))
		dv2 = norm(random_distance_vector(n, weighted))
		d += distance.euclidean(dv1, dv2) ** 2
	return 1.0*d/m


def semi_random_tree_distance(Z, m, weighted=False, norm=zscore):
	"""Calculates the average squared path-difference distance between a given and a random dendrogram.

	Args:
		Z (ndarray): Linkage of the given dendrogram.
		m (int): The number of runs to average over.
		weighted (boolean): True if the weighted distance should be calculated; False if all edges have a weight of 1.
		norm (func): Normalisation for the distance vectors. Not used if weighted=False.
	
	Returns:
		float: The average squared path-difference distance.
	
	"""
	if not weighted:
		norm = lambda x: x
	n = len(Z)+1
	dv1 = norm(distance_vector(Z, weighted))
	d = 0
	for i in range(m):
		dv2 = norm(random_distance_vector(n, weighted))
		d += distance.euclidean(dv1, dv2) ** 2
	return 1.0*d/m


def leaf_distance(node1, node2, weighted):
	"""Calculates the minimal distance between two leaves.

	Args:
		node1 (`Node`): The 1st leaf.
		node2 (`Node`): The 2nd leaf.
		weighted (boolean): True if the weighted distance should be calculated; False if all edges have a weight of 1.
	
	Returns:
		float: The leaf distance.

	"""
	if node1.name == node2.name:
		return 0
	path = 0
	while True:
		path += 1
		parent = node1.parent
		child = (parent.children[0] if parent.children[0] is not node1 else parent.children[1])
		cluster = [(child, 1)]
		while len(cluster) > 0:
			current = cluster.pop(0)
			if current[0].name == node2.name:
				if weighted:
					return 2*parent.distance
				else:
					return path
			cluster += [(child, current[1]+1) for child in current[0].children]
		node1 = parent
	return None
	

def distance_vector(Z, weighted):
	"""Creates a distance vector for a dendrogram.

	Args:
		Z (ndarray): Linkage of the dendrogram.
		weighted (boolean): True if the weighted distance should be calculated; False if all edges have a weight of 1.
	
	Returns:
		list of float: The pairwise leaf distance.
	
	"""
	n = len(Z)+1
	nodes = {i : Node(i, 0.0, []) for i in range(n)}
	for i, row in enumerate(Z):
		node = Node(n+i, row[2], [nodes[int(row[0])], nodes[int(row[1])]])
		nodes[int(row[0])].parent = node
		nodes[int(row[1])].parent = node
		nodes[n+i] = node
	vector = []
	for i in range(n):
		for j in range(n):
			vector.append(leaf_distance(nodes[i], nodes[j], weighted))
	return vector


def random_distance_vector(n, weighted):
	"""Creates a distance vector for a random dendrogram.

	Args:
		n (int): The number of leaves in the dendrogram.
		weighted (boolean): True if the weighted distance should be calculated; False if all edges have a weight of 1.
	
	Returns:
		list of float: The pairwise leaf distance.
	
	"""
	vector = []
	if weighted:
		s = list(np.random.uniform(0,1,int((n*n-n)/2.0)))
		d = {}
		c = 0
		for i in range(n):
			for j in range(n):
				if i < j:
					vector.append(s[c])
					d[(i, j)] = s[c]
					c += 1
				elif i > j:
					vector.append(d[(j, i)])
				else:
					vector.append(0.0)
	else:
		nodes = {i : Node(i, 0.0, []) for i in range(n)}
		clusters = [i for i in range(n)]
		while len(clusters) > 1:
			cluster1 = random.choice(clusters)
			clusters.remove(cluster1)
			cluster2 = random.choice(clusters)
			clusters.remove(cluster2)
			cluster3 = max(nodes.keys())+1
			clusters.append(cluster3)
			node = Node(cluster3, 0, [nodes[cluster1], nodes[cluster2]])
			nodes[cluster1].parent = node
			nodes[cluster2].parent = node
			nodes[cluster3] = node
		for i in range(n):
			for j in range(n):
				vector.append(leaf_distance(nodes[i], nodes[j], False))
	return vector


def distance_vector_from_tree(structure, languages):
	"""Creates an (unweighted) distance vector for a given dendrogram.

	Args:
		structure (list of obj): Dendrogram as nested lists.
		languages (list of str): The languages in the correct oder (for indexing).
			The list must contain exactly those languages which are leaves in the structure.
	
	Returns:
		list of float: The pairwise leaf distance.
	
	"""
	n = len(languages)
	nodes = {i : Node(i, 0.0, []) for i in range(n)}
	clusters = [(structure, None)]
	i = 0
	while len(clusters) > 0:
		cluster = clusters.pop(0)
		if isinstance(cluster[0], list):
			index = 2*n-3-i+1
			i += 1
			node = Node(index, 0.0, [], cluster[1])
			if cluster[1] is not None:
				cluster[1].children.append(node)
			for child in cluster[0]:
				clusters.append((child, node))
			nodes[index] = node
		else:
			index = [j for j, language in enumerate(languages) if language == cluster[0] or language.endswith(", " + cluster[0])][0]
			nodes[index].parent = cluster[1]
			cluster[1].children.append(nodes[index])
	vector = []
	for i in range(n):
		for j in range(n):
			vector.append(leaf_distance(nodes[i], nodes[j], weighted=False))
	return vector