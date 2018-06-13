# BDT
A BDT trained with the same global features as DeepIsolation, but with "annuli" summary variables instead of pf candidates. Intended to serve as a benchmark for DeepIsolation to beat.

### Instructions
1. `source setup.sh`
2. Train BDT: `root -l -b -q "learn.C+(1000000)" > log_file.txt`
3. Make ROC curve from output file: `source ../NeuralNet/setup.sh ; python roc_curve.py`. This creates a plot of the ROC curve and also saves the ROC curve in a `.npz` file so it can be analyzed later along with DNN results.
	1. Add the `TMVA` output files that you want to create ROC curves for on line 10 of `roc_curve.py`.


### Notes
`learn.C` accepts 1 argument: the number of training events. 1 million seems to be more than enough.
You can change the variables that the BDT uses by adding/commenting out lines in `learn.C`.

