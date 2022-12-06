# Reggae: Dipole modes from `pbjam`

We implement a generative model for dipole gravitoacoustic mixed modes using the parameterisation of Ong & Basu (2020). These parameters are fitted to the power spectrum in a fashion analogous to pbjam's "asymptotic peakbagging" step, whereby the generative model is used to construct a model power spectrum which is compared directly to the observational power spectrum, in order to constrain the model parameters. At present, the frequency-dependent coupling strength is described with two parameters (one for each of the two matrices entering into the parameterisation), with a conversion to the asymptotic $q$ provided by an expression in Ong & Gehan (in review, 2022). This expression is in turn used to generate stretched echelle power plots for diagnostic purposes.

In addition to a `DipoleStar` class (analogous to `star` in `pbjam`),

## Contributors

- Joel Ong
- Martin B. Nielsen
- Guy R. Davies
- Emily Hatt