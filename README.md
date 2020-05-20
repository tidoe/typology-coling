# Real-Valued Logics for Linguistic Typology

## Set-Up

Create a new virtual environment and install the required packages. In Linux, this can be done as follows:

```sh
cd typology-coling
python3 -m venv env
source env/bin/activate
pip install numpy==1.17.3
pip install -r requirements.txt
```

Note that the `numpy` package must be installed before the other packages.

## Language Vectors

### From scratch

Define your own language vectors in a matrix:

```python
matrix = [
	# VSO   SVO   SOV   VOS   OVS   OSV  Postp
	[0.11, 0.81, 0.01, 0.01, 0.02, 0.04, 0.04], # English
	[0.00, 0.93, 0.00, 0.00, 0.00, 0.06, 0.01]  # German
]
languages = ["English", "German"]
properties = ["VSO", "SVO", "SOV", "VOS", "OVS", "OSV", "Postp"]
```

### From resources

It is possible to create language-property matrices from external resources with `create_matrix.py`.

```python
from create_matrix import *
```

#### UD treebanks

Create a new directory `typology-coling/ud` and download the Universal Dependencies treebanks from [here](https://universaldependencies.org/#download). In Linux, version 2.5 can be downloaded as follows:

```sh
mkdir ud
cd ud
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105{/ud-treebanks-v2.5.tgz,/ud-documentation-v2.5.tgz,/ud-tools-v2.5.tgz}
cat *.tgz | tar -zxvf - -i
```

If you use a later version than 2.5, you might have to (manually) update the list of UD languages in `typology-coling/files/language_families.txt`.

Create language vectors with single-link, double-link and chain-link properties:

```python
matrixUD, languagesUD, propertiesUD = load_language_vectors("matrices/matrixUD.pickle", name="UD", save_overwrite=True, combine_treebanks=True, treebank_path="ud/ud-treebanks-v2.5/")
```

`combine_treebanks=True` produces language vectors by merging all treebanks for the same language; `combine_treebanks=True` produces treebank vectors. `save_overwrite=True` saves the calculated matrix to the specified file and overwrites it if it already exists; `save_overwrite=False` loads the matrix from the specified file if it exists and calculates it otherwise, but does not save it. To load a matrix from a file if it exists and otherwise calculate and save it, you can write `save_overwrite=(not os.path.exists("matrices/matrixUD.pickle"))`.

#### URIEL ([lang2vec](https://github.com/antonisa/lang2vec))

Create language vectors with syntactic (WALS), phylogenetic and geographic properties:

```python
matrixURIEL, languagesURIEL, propertiesURIEL = load_language_vectors("matrices/matrixURIEL.pickle", name="URIEL", save_overwrite=True, features_sets=["syntax_wals", "fam", "geo"])
```

#### Serva and Petroni (2008)

Create language vectors with conceptual properties (values are strings):

```python
matrixSP, languagesSP, propertiesSP = load_language_vectors("matrices/matrixSP.pickle", name="SP", save_overwrite=True)
```

### Example 1

For instructions concerning

- graphical representation of language vectors
- subselection of languages and properties
- clustering of language vectors
- computation of tree distances

see `example_1.py`.

## Real-Valued Logics

### Valuations

`valuation.py` defines some example instances of the abstract `Valuation` class, e.g. valuations for fuzzy logic and product logic. There are also some application examples at the bottom of the script. Therefore, only one example is repeated here.

First, get the property vectors from the language vectors, which is basically transposing the matrix:

```python
matrix_T = np.transpose(np.array(matrix, dtype=np.float64))
property_vectors = {properties[i] : v for i, v in enumerate(matrix_T)}
```

With the example from above, this yields:

```python
property_vectors = {
	          # English  German
	"VSO"   : [ 0.11,    0.00 ],
	"SVO"   : [ 0.81,    0.93 ],
	"SOV"   : [ 0.01,    0.00 ],
	# ...
	"Postp" : [ 0.04,    0.01 ]
}
```

Instantiate a valuation, e.g. `VFuzzy`, with the property vectors:

```python
from valuation import VFuzzy

valuation = VFuzzy(property_vectors)
```

Define and parse logical formulae (spaces around operators and brackets are important):

```python
from formula_parser import parse_formula

formula1 = "SVO ⇔ ( ¬ Postp )"
formula2 = "SOV ⇔ ( ¬ Postp )"
term1 = parse_formula(formula1)
term2 = parse_formula(formula2)
```

Supported connectives are ¬ (negation), & (conjunction), | (disjunction), ⇒ (implication), ⇔ (equivalence) and + (addition).

Evaluate the formulae and calculate the average truth values:

```python
valuation.evaluate(term1)
print(valuation.collapse()) # 0.870
valuation.evaluate(term2)
print(valuation.collapse()) # 0.025
```

### Example 2

For a full example, including phylogenetic weighting, see `example_2.py`.

# COLING

To reproduce the results of the papers published at COLING, set-up the virtual environment and download the UD treebanks as described above. Then run the following commands:

```sh
# Differences in dependency direction
python run_inconsistencies.py

# Evaluate Greenberg's universals on the UD treebanks
python run_greenberg.py 1

# Run six random-split experiments
python run_greenberg.py 2

# List of implications
python run_implications.py
```
