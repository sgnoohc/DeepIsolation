# DeepIsolation
Deep neural network for identifying prompt vs. fake leptons

## Instructions
1. `git clone https://github.com/sam-may/DeepIsolation ; cd DeepIsolation`
2. Make baby `root` files that can be used to train BDT and DNN. See instructions in `BabyMaker` directory.
3. Train BDT (see instructions in `BDT` directory) and/or DNN (see instructions in `NeuralNet` directory).
4. Evaluate results (see instructions in `NeuralNet` directory). 

## Singularity shell command
singularity shell --bind /usr/lib64/nvidia:/host-libs /cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow-gpu:latest
