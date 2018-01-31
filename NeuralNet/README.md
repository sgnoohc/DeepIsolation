# NeuralNet
Deep neural network for identifying prompt vs. fake leptons

### Running a small test case
1. Convert baby from `ROOT` to `hdf5` file: `prep.py <path_to_baby>.root` 
2. Invoke singularity container to run on GPU (on uaf-1): `singularity shell --bind /usr/lib64/nvidia:/host-libs /cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow-gpu:latest` , then `bash`
3. Train: `python3 learn.py <output_filename> <nTrainingEvents>`
