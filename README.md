# DeepCorrectionsExperiments.jl

Scripts to reproduce the experiments from the paper "Decomposition Methods with Deep Correctionsfor Reinforcement Learning" by M. Bouton, K. Julian, A. Nakhaei, K. Fujimura, and M. J. Kochenderfer, 2019.

## Usage

**Training a network:**

To train a network use the script `crosswalk_train.jl`.

Run `julia crosswalk_train.jl --help` to display all the available options. 

- `--single` : set the number of pedestrians to 1. It will train a policy on the subtask
- `--vdn`: train a value decomposition network
- `--seed`: specify the random seed (type: Int64, default: 1)
- `--training_steps` specify the number of training steps (type: Int64, default: 10000)
- `--eps_fraction` Fraction of the training set to use to decay epsilon from 1.0 to eps_end, this is the epsilon used in the epsilon-greedy exploration policy (type: Float64, default: 0.5)
- `--eps_end` epsilon value at the end of the exploration phase (type: Float64, default: 0.01)
- `--correction` specify the file with the single policy to load  (type: Union{Nothing, String})
- `--weight` weight of the correction network (type: Float64, default: 0.1)
- `--logdir` Directory in which to save the model and log training and evaluation data (default: "log")
- `--n_eval` Number of episodes for evaluation (type:Int64, default: 1000)

To train a correction network you must specify the folder with a policy trained on the single pedestrian problem. 
The script will train a network and evaluate its performance. 

To change the network architecture you may edit the file `crosswalk_train.jl`

**Evaluating a pre-trained model:**

If you wish to run evaluation only you can use the `crosswalk_eval.jl` script.

**Visualizing the results** 

The script `plot_performance.jl` can be useful to visualize the content of the `csv` files with the evaluation data.

## Installation 

Requires julia 1.1+. Copy paste the following lines in the julia REPL to install the necessary dependencies:

```julia 
using Pkg
Pkg.add("POMDPs")
using POMDPs
POMDPs.add_registry()

# install automotive simulator
Pkg.add(PackageSpec(url="https://github.com/sisl/Vec.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/Records.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutomotiveDrivingModels.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutoViz.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutoUrban.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutomotiveSensors.jl"))
Pkg.add(PackageSpec(url="https://github.com/JuliaPOMDP/RLInterface.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutomotivePOMDPs.jl"))

# add the deep corrections package
Pkg.add(PackageSpec(url="https://github.com/sisl/DeepCorrections.jl"))

# add this repo
Pkg.add(PackageSpec(url="https://github.com/MaximeBouton/DeepCorrectionsExperiments.jl"))