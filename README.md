[![Unit tests](https://github.com/gcicconofri098/CMEPDA_project/actions/workflows/unittest.yml/badge.svg)](https://github.com/gcicconofri098/CMEPDA_project/actions/workflows/unittest.yml)
<a href='https://neutrinos-icecube.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/neutrinos-icecube/badge/?version=latest' alt='Documentation Status' />
</a> 

# CMEPDA_project
Repository for the project for the CMEPDA exam. The project consists of studying the IceCube experiment Monte Carlo simulations, particularly on predicting the original trajectory of the simulated particles using Machine Learning (ML) techniques.

# Synopsis

The IceCube experiment is located underground in Antarctica, and it consists of an array of photomultipliers tubes (PMTs) which detect the Cherenkov light that particles (such as muons, neutrinos and so on) emit as they traverse the ice where IceCube is located. The data from the PMTs (location, time, charge) is then used for predicting the original direction of the particle.

Usually, analysis is made with analytic fits and likelihood functions that take into account the morphology of the ice the experiment is contained in, however, newer approaches that use ML techniques are becoming a possible alternative.

# Dataset

The dataset was taken from the Kaggle competition: 

IceCube - Neutrinos in Deep Ice (https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/overview),

and consists of a number of events that record the hits inside the detectors and relative information.

# Documentation

The documentation can be found at the following link: http://neutrinos-icecube.readthedocs.io/
