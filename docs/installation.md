# Installation

Reggae requires the following packages to be installed:

- Basics: `numpy`, `matplotlib`, and `dill`
- Astronomy-specific: `lightkurve`, `astropy`, and `pbjam`
- Numerics: `dynesty` and `jax`
- Mixed-mode asteroseismology: the `zeta` package (not published on PyPI owing to limited scope)

In addition, to operate the Qt GUI console, PyQt has to be installed. Preferably, this should be done using the system package manager or `conda`, rather than `pip`, since it links against compiled Qt libraries.