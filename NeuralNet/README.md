# NeuralNet
Deep neural network for identifying prompt vs. fake leptons

### Instructions 
1. Convert babies from `ROOT` to `hdf5` file: `python prep.py` 
2. Invoke singularity container to run on GPU (on uaf-1): `singularity shell --bind /usr/lib64/nvidia:/host-libs /cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow-gpu:latest` , then `bash`
3. Train: `python3 learn.py <output_filename> <nTrainingEvents>`
4. Compare performance to BDT and RelIso: `python makePlots.py` (you likely will need to make changes to the input files)
