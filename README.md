# PARENT
Code for [Handling Divergent Reference Texts when Evaluating Table-to-Text Generation](https://arxiv.org/abs/1906.01081).

This largely follows [the code given by the google-research team](https://github.com/google-research/language/tree/master/language/table_text_eval), all credit to them. I simply have rewritten some part to be faster, removed all tensorflow mentions and wraped everything using multiprocessing (shipped with python).

Original code can take up to several minutes to compute scores on the WikiBIO test set. With this implementation, it takes less than 10 seconds with 32 cpus.

**EDIT 06-04-2021**: I have package everything so that it can be installed with pip.
That way, you can install the repo once for all your projects. TODO: push to PyPI

```bash
git clone https://github.com/KaijuML/parent.git
cd parent
pip install .
```

I have also included better support for multi references. Now, references can be:

- A list of single references. We have `len(references) == len(predictions)` and `references[k]` is a list of str
- A list of multiple references, gathered by example. We have `len(references) == len(predictions)` and `references[k]` is a list of references, which are lists of str
- A list of multiple references, gathered by reference. We have `len(references) == <max_number_of_refs_for_an_example` and `references[k]` is the list of kth references for all examples. If an example has less than k refs, then `references[k][ex] == ''`

**EDIT 28-01-2021**: I have added support for multiple references. Simply pass a list of files with: `--references <file1> <file2> ... <fileN>`.  
`<file1>` should contain the first reference for all instances (and therefore should have no empty line.) `<file2>` should contain the second reference for all instances (if an instance does not have a second ref, there should be an empty line instead). So on, so forth for `<fileN>`.

Note that for simplicity, I make a very simple and naive check to see if multiple instances are passed (see `parent.py:line347-355`). This could easily break in edge-case settings (e.g. when the code is called on files with only one instance).


## Computing the PARENT score in command line:

If you want out-of-the-box usage, simply use:

```parent --predictions $PREDICTION_PATH --references $REFERENCES_PATH --tables $TABLES_PATH --avg_results```

With the example files provided in `data`, and using `--n_jobs 32`, this should take around 8 secondes and print:

```
PARENT-precision: - - - 0.797
PARENT-recall:  - - - - 0.45
PARENT-fscore:  - - - - 0.553
```

In comparison, running the original script takes around 1m40s and returns `Precision = 0.7975 Recall = 0.4503 F-score = 0.5529`

### File format

Note that predictions/references should be one sentence per line (whitespace is used to tokenize sentences).

Tables should be in a json-line file, with one table per line in json format (```json.loads(line)``` will be called).


## Computing the PARENT score in a notebook:

You can also use the code anywhere, simply follow this example:

```python
from parent import parent
import json


# open all files
path_to_tables = 'data/wb_test_tables.jl'
path_to_references = 'data/wb_test_output.txt'
path_to_predictions = 'data/wb_predictions.txt'

with open(path_to_tables, mode="r", encoding='utf8') as f:
    tables = [json.loads(line) for line in f if line.strip()]

with open(path_to_references, mode="r", encoding='utf8') as f:
    references = [line.strip().split() for line in f if line.strip()]

with open(path_to_predictions, mode="r", encoding='utf8') as f:
    predictions = [line.strip().split() for line in f if line.strip()]
        
precision, recall, f_score = parent(
    predictions,
    references,
    tables,
    avg_results=True,
    n_jobs=32,
    use_tqdm='notebook'
)
```
