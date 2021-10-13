# Robustness Certification for Point Cloud Models <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="https://www.sri.inf.ethz.ch/assets/images/sri-logo.svg"></a>

![Overview](overview.png)

3DCertify is the first verifier to certify the robustness of 3D point cloud models against semantic, real-world transformations and point perturbations.
It leverages first-order Taylor approximations to efficiently compute linear relaxations of any differentiable transformation function.
These relaxations can be used to certify a network's robustness to common transformations such as rotation, shifting, or twisting, using a state-of-the-art general network verifier as a backend (e.g., [ERAN](https://github.com/eth-sri/eran) or [auto_LiRPA](https://github.com/KaidiXu/auto_LiRPA)).
Furthermore, 3DCertify improves these certifiers with a tighter relaxation for max pool layers, which are particularly challenging and a crucial component of point cloud architectures.

This repository contains all implementations, models, and instructions required to reproduce the experiments from our ICCV'21 paper.
Please refer to the paper for the theoretical introduction and analysis, as well as detailed results.

Paper Links: &emsp; [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Lorenz_Robustness_Certification_for_Point_Cloud_Models_ICCV_2021_paper.html) &emsp; [ArXiv](https://arxiv.org/abs/2103.16652)

This work is part of the [SafeAI](http://safeai.ethz.ch) project at the [SRI lab](https://www.sri.inf.ethz.ch) of [ETH Zurich](https://ethz.ch/).

## Setup Instructions

Clone this repository, including all submodules:

```bash
git clone --recurse-submodules https://github.com/eth-sri/3dcertify.git
```

Create a [conda](https://www.anaconda.com/products/individual) environment with the required dependencies:

```bash
conda env create -f environment.yml
conda activate 3dcertify
```

Setup dependencies and install ERAN:

```bash
cd ERAN
# select an ELINA version compatible with the selected version of ERAN
sed -i'' -e '/^cd ELINA$/a git checkout 2c9a4ea' install.sh
./install.sh
cd ..
```

For DeepG3D relaxations, install Deepg3D (see also `deepg3d/README.md`):

```bash
cd deepg3d/code
mkdir build
make deepg_pointclouds
cd ../..
```

For experiments using auto_LiRPA, install auto_LiRPA (see also `auto_LiRPA/README.md`):

```bash
cd auto_LiRPA
python setup.py develop
git apply ../auto_LiRPA.diff
cd ..
```   

DeepG3D and some parts of DeepPoly use the GUROBI solver for certification. To run our code, apply for and download
an [academic GUROBI License](https://www.gurobi.com/academia/academic-program-and-licenses).

## Run Certification

All experiments from our paper can be reproduced by running the commands listed below with appropriate parameters. Some
basic pretrained models are provided with this repository in `models/`, all additional models used in our experiments
can be downloaded at https://files.sri.inf.ethz.ch/pointclouds/pretrained-models.zip or via the script in
`models/download_models.sh`. Alternatively, you can train you own models using our training scripts `train_*.py`.

_Note: the datasets will be downloaded and processed automatically the first time the script is used. Depending on
processing power, this may take several hours._

### Semantic Transformations using Taylor3D

```bash
python verify_transformation.py \
    --model models/64p_natural.pth \
    --num_points 64 \
    --transformation RotationZ \
    --theta 1deg \
    --intervals 1 \
    --relaxation taylor \
    --pooling improved_max \
    --experiment example1
```

Available transformations: `RotationX`, `RotationY`, `RotationZ`, `TwistingZ`, `TaperingZ`, `ShearingZ`

For automatic composition of arbitrary transformations chain them using a + symbol, e.g. `RotateZ+RotateX` or
`TaperingZ+TwistingZ+RotationZ`.

A detailed description of all parameters and their possible values can be accessed via the integrated help
`python verify_transformation.py -h`.

### Point Perturbation

```bash
python verify_perturbation.py \
    --model models/64p_ibp.pth \
    --num_points 64 \
    --eps 0.01 \
    --pooling improved_max \
    --experiment example2
```

### Semantic Transformations using DeepG3D

Compute relaxations with DeepG3D:

1. Create a directory with a config file with parameters and relaxations, such as
   `deepg3d/code/examples/modelnet40_64p_rotationz_theta_1_intervals_1/config.txt`. Refer to the
   [Deepg3D README](deepg3d/README.md) for more information.
2. enter the `deepg3d/code` directory and run
   `.build/deepg_pointclouds examples/modelnet40_64p_rotationz_theta_1_intervals_1`

Then verify using 3DCertify:

```bash
python verify_deepg.py \
    --model models/64p_natural.pth \
    --spec-dir deepg3d/code/examples/modelnet40_64p_rotationz_theta_1_intervals_1 \
    --num_points 64 \
    --pooling improved_max \
    --experiment example3
```

### Part Segmentation

```bash
python verify_segmentation.py \
    --model models/64p_segmentation.pth \
    --num_points 64 \
    --transformation RotationZ \
    --theta 1deg \
    --intervals 1 \
    --relaxation taylor \
    --experiment example4
```

### Transformation using auto LiRPA

```bash
python verify_lirpa.py \
    --model models/64p_natural.pth \
    --num_points 64 \
    --theta 1deg \
    --experiment example5
```

## Citing this Work

```
@inproceedings{lorenz2021robustness,
    author       = {Tobias Lorenz and
                    Anian Ruoss and
                    Mislav Balunovi{\'c} and
                    Gagandeep Singh and
                    Martin Vechev},
    title        = {Robustness Certification for Point Cloud Models},
    year         = 2021,
    month        = {October},
    booktitle    = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}
    pages        = {7608-7618}
}
```

## Contributors

* [Tobias Lorenz](https://www.t-lorenz.com) (tobias.lorenz@cispa.de)
* Anian Ruoss (anian.ruoss@inf.ethz.ch)
* [Mislav Balunović](https://www.sri.inf.ethz.ch/people/mislav) (mislav.balunovic@inf.ethz.ch)
* [Gagandeep Singh](https://ggndpsngh.github.io/) (ggnds@illinois.edu)
* [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin) (martin.vechev@inf.ethz.ch)

## License

Licensed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
