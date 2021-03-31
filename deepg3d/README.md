# DeepG3D <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

DeepG3D is our generalization of the original [DeepG](https://github.com/eth-sri/deepg/) implementation for images to
compute linear bounds for semantic transformation of 3D point clouds. It is distributed as a part
of [3DCertify](https://github.com/eth-sri/3dcertify/), the first certification system for point cloud models.

## Setup instructions

We assume you have already installed the Gurobi solver as part of the ERAN installation from the main setup
instructions. If not, please install it first and make sure the related environment variables `GUROBI_HOME`, `PATH`
and `LD_LIBRARY_PATH` point to the correct locations.

Then, you should be able to compile DeepG3D:

```
cd code
mkdir build
make deepg_pointclouds
```

## Example for constraint computation

As an example, we will compute constraints for the ModelNet40 dataset for rotation between -1 and 1 degrees. These
constraints can then be used to certify robustness using 3DCertify (example in main README.md).

```bash
cd code
./build/deepg_pointclouds examples/modelnet40_64p_rotationz_theta_1_intervals_1_eps_0.0000001
```

In order to reproduce the experiments from our paper, you can run the example configurations provided in
`code/examples`. We provide configurations for all of our mayor findings - other experiments can be reproduced by simply
adjusting the appropriate parameters.

## Format of configuration file

Each experiment directory should have config.txt file which containts the following information:

```
dataset               dataset which we are working with, should be one of {modelnet40, shapenet}
chunks                number of splits along each dimension (each split is certified separately)
inside_splits         number of refinement splits for interval bound propagation
method                method to compute the constraints, to use DeepG it should be set to polyhedra
spatial_transform     description of geometric transformation in predefined format, see the examples
num_tests             number of images to certify
ub_estimate           estimate for the upper bound, usually set to Triangle
num_attacks           number of random attacks to perform for each image
poly_eps              epsilon tolerance used in Lipschitz optimization in DeepG
num_threads           number of threads (determines for how many pixels to compute the constraints in parallel)
max_coeff             maximum coefficient value in LP which DeepG is solving
lp_samples            number of samples in LP which estimates the optimal constraints
num_poly_check        number of samples in sanity check for soundness
set                   whether to certify test or training set
```

Parameter name and value are always separated by spaces. You can look at provided configuration files in `code/examples`
to see the values used in our experiments.
