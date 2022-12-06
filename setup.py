import setuptools
import os

version = {}

with open(os.path.join(*['reggae','version.py'])) as fp:
	exec(fp.read(), version)

setuptools.setup(
    name="reggae",
    version=version['__version__'],
    author="Joel Ong, +",
    author_email="joelong@hawaii.edu",
    description="Dipole modes for pbjam",
    packages=['reggae'],
    install_requires=open('requirements.txt').read().splitlines(),
    #extras_require={'docs': ["nbsphinx"]},
    #include_package_data=True,
    #license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha"
    ],
)
