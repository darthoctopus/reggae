# Installation

**We presently only officially support installating Reggae under Python 3.10 and below**. Before installing Reggae we strongly recommend you use a virtual environment. The simplest way to set one up is using the Python `venv` module:
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

## Testing things work

To check that everyone works you can try running Reggae with an example provided in the test/jamie directory
```
cd path/to/my/repos/reggae/test/jamie
python test.py
```
This should open up an UI featuring the example data.

## Installation FAQs

- What do I need to install in order to accelerate these calculations on my GPU?

Please check that your OS is on `jax`'s supported list of systems and hardware configurations, which can be found [here](https://jax.readthedocs.io/en/latest/installation.html#supported-platforms). In summary, while performing `jax` calculations on the CPU is supported on all platforms, GPU acceleration is only available on limited combinations of hardware and operating systems. Our `requirements.txt` file assumes only CPU support via `jaxlib`; Windows users may need additional external requirements for it to work.