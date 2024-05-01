# Usage

Reggae is intended to be operated primarily through its GUI, although many GUI actions correspond to methods of the {class}`~reggae.dipolestar.DipoleStar` class. There are a couple of ways to launch the GUI:

- Executing the `reggae` module from the command line: `python -m reggae`. This will launch an empty instance of the GUI.
- Instantiating a {class}`~reggae.qtconsole.ReggaeDebugWindow`. Passing it no arguments will launch an empty instance, while passing it a `pbjam.star.star` or {class}`~reggae.dipolestar.DipoleStar` object will open that object in the GUI.

## Diagnostic Information

As a diagnostic utility, Reggae exposes a large amount of information to the viewer. This information is broken down in to several panels, which are switchable with the tab bar on the left side of the screen.

:::{figure} ../screenshots/ps.png
:alt: Left Panel 1: Power Spectrum

Left Panel 1: Power Spectrum
:::

The first panel shows the power spectrum. Like PBJam, Reggae uses the so-called SNR power spectrum (i.e. with a smooth background divided out) as its focus is on fitting individual peaks — this is shown in black. To further focus on the dipole modes, a model of the radial and quadrupole modes inherited from PBJam is further divided out — this is shown in blue. Thus, any residual peaks are those ignored by PBJam, which, a priori, are primarily dipole modes. However, the user should ensure that PBJam's model works as expected before proceeding further. The black and blue curves should agree except at the locations of PBJam's identified radial and quadrupole modes, at which locations there should be exactly two peaks at a time in the black curve that are not reflected in the blue.

Later steps in the fitting procedure produce PSD models as well. These are shown with various coloured curves.

:::{figure} ../screenshots/echelle.png
:alt: Left Panel 3: Echelle Diagrams

Left Panel 3: Echelle Diagrams
:::

Given that the p- and g-modes are known to separately possess characteristic frequency and period spacings, respectively, it is conventional to examine the locations of these peaks in the power spectrum through the use of "echelle power diagrams". This third panel shows such diagrams, where the power spectrum is phase-folded with respect to these frequency or period spacings on the horizontal axis.

Reggae's mode identification step turns a small number of parameters — such as $\Delta\Pi$, $\epsilon_g$, coupling strengths, and core/envelope rotation rates — into a large number of mode frequencies and heights, each one corresponding to a peak in the power spectrum associated with a mode of mixed p-like and g-like character. A good set of Reggae's parameters should also result in mode frequencies, placed on these diagrams, that align visually with local maxima in the power spectrum.

For the period-echelle power diagram in particular, the horizontal coordinate is further "stretched" through the use of the coordinate transform described in Mosser et al. 2012, 2015, 2017, 2018; Ong & Gehan 2023. In summary: in the asymptotic regime, mixed modes satisfy an eigenvalue equation parameterised by a single coupling strength $q$. If ansatz values of $q$, $\Delta\Pi$, and the pure p-mode frequencies are supplied, this allows the pure g-mode frequencies to be inferred from the mixed modes analytically. More generally, this specifies a coordinate transformation in the period coordinate mapping the mixed mode frequencies to the g-mode frequencies. As such, when applied to the period-echelle power diagram, it will map avoided crossings to vertical ridges, if the supplied values of $q$, $\Delta\Pi$, and the p-mode frequencies are accurate.

## Tuning Parameters

Reggae is intended to permit the user to derive estimates of parameters describing the dipole modes. As such, its GUI provides various ways by which one might either manually fine-tune a guess at these parameters, or else arrive at point esimates or posterior distributions through automated optimisation or sampling. These options are shown on the right side of the screen. Again, they are broken down into several panels, which are switchable with the right tab bar.

:::{figure} ../screenshots/echelle.png
:alt: Right Panel 1: Manual Tuning

Right Panel 1: Manual Tuning
:::