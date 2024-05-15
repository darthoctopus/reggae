# Installation
Before installing Reggae we strongly recommend you use a virtual environment. The simplest way to set one up the Python venv module:
```
python -m /path/to/new/virtual/environment
``` 
You will now need to activate the environment before you can install new Python packages. To do so just do:
```
source /path/to/new/virtual/environment/bin/activate
```
You can now freely install new packages without interfering with your previously installed packages.

To install reggae do the following:
 
- Next clone the `zeta` repository and install it by doing the following 
```
cd path/to/my/repos
git clone https://gitlab.com/darthoctopus/zeta.git
cd zeta
pip install -e .
```
- Now clone the `Reggae` repository and install it by doing the following
```
cd path/to/my/repos
git clone https://github.com/darthoctopus/reggae.git
cd reggae
pip install -e .
```
That should be it!

