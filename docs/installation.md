# Installation
To install reggae do the following:
- First make sure you have either PyQT 5 or 6 installed. Reggae relies on the Qt GUI console and so PyQt has to be installed. Preferably, this should be done using your system package manager or `conda`, rather than `pip`, since it links against compiled Qt libraries.
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

