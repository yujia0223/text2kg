## Results

Contained here are the output of the benchmark_and_evaluate script for most of our models. The "tests" and "evaluation" folders were used for testing when updating the evaluation code and for legacy results respectively. "old_results" contains results files from the old version of the evaluation code. Some older results may exist in the main folder, so if checking any of these files gives different results than were documented, it may be the case that an older file exists here. This can be easily checked by running the raw_output of a model through the evaluation code again, which for most models only takes a couple of minutes.

There are two types of results files, the normal files that only contain the scores and counts of correct, incorrect, missed, partial, spurious, actual and possible. The scores in the files can be printed out to the terminal window using:

```python get_eval_results.py --file=PATH```

The other type are the detailed results files, which in addition to the scores contain the input text, reference triples and candidate triples. These can be examined line by line using:

```python extract_det.py --det_file=PATH```