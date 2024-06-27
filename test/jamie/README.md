# Reggae Test Data: KIC 9267654

This test data set is generated from having run pbjam 1.0.3 on KIC 9267654 (Tayar et al. 2022, ApJ, 940, 23), a red giant whose surface rotation rates, separately estimated from spectroscopic V sin i and from l=2 and l=3 p-mode rotational splittings, appear to disagree. By explicitly accounting for near-degeneracy effects, reggae also permits an independent estimate of envelope rotation, penetrating more deeply than quadrupole and octopole modes, to be derived from only dipole mixed modes (Ong, Ahlborn, Tayar, et al., in prep.).

To bring up the reggae GUI, run the file `test.py`. The best manual parameters obtained from hand-tuning Reggae are stored in `session.pkl`, which can be loaded using the "Load Session" button in the right toolbar. Please see our documentation for more details about the other GUI options.

This test file will load the GUI into the state it was at when the manual fitter decided that the current guesses at the dipole mode parameters were "good enough" to be used for other purposes.