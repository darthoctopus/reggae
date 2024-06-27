# Quick Start

Here is a series of steps intended to get you started using Reggae productively, as quickly as possible.

## Exporting output from PBJam

Reggae is intended to be operated using the results of PBJam's peakbagging procedure as a starting point. Ideally, you will already have run `peakbag`, which is a method of the `pbjam.star` object, on a power spectrum of your choice. If you wish to save this for use with a script that relies only on Reggae, it's generally a good idea to save this to a file. For instance:

```
# let's say my_star is an instance of pbjam.star.

import dill

with open("pbjam_result.pkl", 'wb') as f:
	dill.dump(my_star, f)
```

The data structures of PBJam version 2 and higher are completely revamped, so as not to rely on pymc3. Reggae also accepts a `pbjam.modeID.modeID` object from newer versions supporting this interface.

## Creating the {class}`~reggae.dipolestar.DipoleStar` Object

There are two ways in which one may do this. One might first start the GUI, and select a pickle file containing a `pbjam.star` object, to load it into working memory. Alternatively, one might directly create the {class}`~reggae.dipolestar.DipoleStar` object using the `from_pbjam` class method, and then launch the GUI console later by passing it as a positional argument when initialising a {class}`~reggae.qtconsole.ReggaeDebugWindow`.

## Using the GUI

Once the {class}`~reggae.dipolestar.DipoleStar` object has been loaded into a GUI session, you may now modify values of $\delta_{01}$, $\Delta\Pi_0$, $p_L$, and $p_D$. See [our description of the GUI itself](project:usage.md) for how to change specific properties of the PSD model. There isn't a well-defined procedure for doing this (yet). However, you may find the following pointers helpful:

- There is a tight empirical $\Delta\nu$-$\Delta\Pi_1$ sequence for first-ascent red giant stars; see e.g. fig. 1 of [Deheuvels et al. 2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...659A.106D/abstract). More evolved (e.g. core-helium-burning aka CHeB) stars lie on a different, and less well-understood, part of this diagram (e.g. fig. 6 of [Rui et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.1618R/abstract)). If one can identify the kind of red giant one is dealing with, one may convert a putative value of $\Delta\Pi_1$ to an initial guess for `DPi0` as

$$\texttt{DPi0} = \sqrt{2}(\Delta\Pi_1/125\ \mathrm{s})$$

- For first-ascent red giants, there is a known, if rough, empirical relation between the coupling strength $q$ and the quantity $\mathcal{N}_1 = \Delta\nu / \nu_\text{max}^2 \Delta\Pi_1$. One may find the value of $q$ associated with the current values of $p_L$ and $p_D$ by using the "Auto-q" button on the GUI. More evolved, or anomalous, stars may not respect this relation, and the conversion between $q$ and these matrix parameters is less accurate with strong coupling (mostly CHeB or subgiant stars).

- Asymptotic analysis suggests that $\epsilon_g$ takes values close to 0.75, although we expect empirically some variation owing to possible internal structural features of the g-mode cavity (e.g. [Vrard et al. 2022](https://ui.adsabs.harvard.edu/abs/2022NatCo..13.7553V/abstract), [Lindsay et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...931..116L/abstract)).

- Roughly speaking, $d_{01}$ takes small, positive, values (less than 0.2, depending on whether or not the peakbagged or asymptotic even-degree modes are used for mode identification).

Once initial guesses for these parameters have been supplied to the GUI, we recommend that you first tweak them until the predicted locations of mode frequencies in the echelle-diagram panel line up with the positions of peaks in the actual echelle power diagram. Once this has been achieved, the goodness of fit of the PSD model may be iteratively improved by optimizing the likelihood function in one variable at a time, keeping all the others fixed. We recommend finding a good value of the normalisation factor first before beginning to tweak the others.

When you are satisfied with your manually tuned values of these asymptotic parameters, you may then use more sophisticated techniques (numerical optimisation or nested sampling), which we expose through the GUI if you prefer to operate through it. Alternatively, these commonly-used routines are also implemented as methods of the {class}`~reggae.dipolestar.DipoleStar` object, if you should prefer for long-running tasks to be run in standalone scripts.

## Saving Your Work

We recommend that you save your work from time to time. The "Dump Session" action in the GUI toolbar will save the current state of the GUI console to a pickle file, while the "Dump Reggae" action will save the currently open {class}`~reggae.dipolestar.DipoleStar` object, including potentially any results from optimisation and/or nested sampling.