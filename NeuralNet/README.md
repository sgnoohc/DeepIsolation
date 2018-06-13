# NeuralNet
Deep neural network for identifying prompt vs. fake leptons

### Instructions 
1. Convert babies from `ROOT` to `hdf5` file: `python prep.py` 
2. Invoke singularity container to run on GPU (on uaf-1): `singularity shell --bind /usr/lib64/nvidia:/host-libs /cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow-gpu:latest` , then `bash`
3. Train: `python3 learn.py <output_filename> <nTrainingEvents>`. This will print out some diagnostic info (AUC, etc.) at the end and will also produce an AUC vs. Epoch curve (`convergence.pdf`).
4. Compare performance to BDT and RelIso: `python makePlots.py` (you likely will need to make changes to the input files)

### Notes
The DNN itself is implemented in `model.py`. You can pick a different model, or create a new one yourself and use it in training by modifying this line of `learn.py`: `https://github.com/sam-may/DeepIsolation/blob/master/NeuralNet/learn.py#L66`.
