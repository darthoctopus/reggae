# Installation

**We presently only officially support installing Reggae under Python 3.10 and below**. Before installing Reggae we strongly recommend you use a virtual environment. The simplest way to set one up is using the Python `venv` module:
```
python -m /path/to/new/virtual/environment
``` 
You will now need to activate the environment before you can install new Python packages. To do so just do:
```
source /path/to/new/virtual/environment/bin/activate
```
You can now freely install new packages without interfering with your previously installed packages.

To install `reggae`, clone this repository and install it by doing the following:
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

- I am getting a package conflict for a module named `zeta` when attempting to install.

`reggae` relies on a package named `zeta` for asymptotic mode-coupling calculations and generation of stretched period-echelle power diagrams. Unfortunately, that name is already taken on PyPI, so it is not available there and has to be installed from its gitlab repository. If installation via `requirements.txt` does not work, it may also be installed manually with the following commands:

```
cd path/to/my/repos
git clone https://gitlab.com/darthoctopus/zeta.git
cd zeta
pip install -e .
```

- I am running into core dump errors with missing libraries.

This is an issue we have encountered on linux systems where only X11 GUI sessions are installed. You will need to use your OS's package manager to install certain X11 libraries that may be required by Qt for the GUI to function properly. For example, if you should see the error message

```
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin.
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
```

then (e.g. if you are using Ubuntu) you will need to install additional system packages:

```
$ sudo apt install libxcb-cursor0
```

As this specific error message suggests, downgrading to Qt 6.5 (and PyQt 6.5) may be an alternative resolution to this issue. Yet another alternative may be to use a Wayland-based GUI session (and correspondingly a different Qt platform plugin), although this may cause other issues.