# xview2

A solution to the 2019 xView2 competition

## Conda Environment

### Setup

Run the following to set up the Conda environment:
```
conda env create -f environment.yml
```
Then activate with `conda activate xview2`.

### Updating

After adding a package to the current Conda environment, update `environment.yml` by running:
```
conda env export --from-history > environment.yml
```
Then commit and push changes.

### GPU

The `environment-gpu.yml` file should be used on machines with with a GPU. It creates the environment `xview2-gpu`.