# PARENT
Code for [Handling Divergent Reference Texts when Evaluating Table-to-Text Generation](https://arxiv.org/abs/1906.01081).

This largely follows the code given by the google-research team, all credit to them. I simply have rewritten some part to be faster, removed all tensorflow mentions and wraped everything using multiprocessing (shipped with python).

Original code can take up to several minutes to compute scores on the WikiBIO test set. With this implementation, it takes only 10 seconds with 32 cpus.

Slight change in functionnality: I have (for now) removed spport for multiple references. This is because I feel this metric is especially usefull on [WikiBIO](https://github.com/DavidGrangier/wikipedia-biography-dataset) and there is only when reference per instance. Support for multiple references could be easily added with a for loop, in method `parent_instance_level`.


### Computing the PARENT score in command line:

If you want out-of-the-box usage, simply use:

```python parent.py --predictions $PREDICTION_PATH --references $REFERENCES_PATH --tables $TABLES_PATH --avg_results```

Note that predictions/references should be one sentence per line (whitespace is used to tokenize sentences).

Tables should be in a json-line file, with one table per line in json format (```json.loads(line)``` will be called).


### Computing the PARENT score in a notebook:

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

with open(path_references, mode="r", encoding='utf8') as f:
    references = [line.strip().split() for line in f if line.strip()]

with open(path_to_predictions', mode="r", encoding='utf8') as f:
    predictions = [line.strip().split() for line in f if line.strip()]
        
precisions, recalls, f_scores = parent(
    predictions,
    references,
    tables,
    avg_results=True,
    n_jobs=32
)
```
