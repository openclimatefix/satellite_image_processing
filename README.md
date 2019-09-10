# Installation

```
git clone
conda env create -f environment.yml
conda activate sat_image_processing
```


### Install Jupyter lab interactive plotting for matplotlib

See the [jupyter-matplotlib docs for more info](https://github.com/matplotlib/jupyter-matplotlib).  The short version is to run these commands from within the `sat_image_processing` env:

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib
```
