# HPvanRhee_MScThesis_Documentation
This repository contains all relevant coding and modelling aspects for my Systems &amp; Control MSc Thesis about MPC based FFR in the 2035 Dutch power system.

SETUP:
- For PFlow_2021 and PFlow_2035: 
    python version      3.10.19 
    numpy version       1.23.5
    matplotlib version  3.10.8
    pandas version      1.5.3
    scipy version       1.9.3
    andes version       1.10.0
    pandapower version  2.10.0
- See environment_PFlow.yml for the full environment
    
- For all other python code: 
    python version      3.10.19 
    numpy version       2.2.6
    matplotlib version  3.10.8
    pandas  version     2.3.3
    scipy version       1.13.1
    andes version       1.10.0
    cvxpy version       1.7.5
    control version     0.10.2
- See environment.yml for the full environment


- For MATLAB code:


After installing ANDES, copy the contents inside of the MPC_code/renewable directory into your local package installation under andes/models/renewable. Make sure to also replace the __init__.py file inside andes/models by the __init__.py file inside MPC_code. After doing this, rung the cell below to prepare the MPC controllers for use

CONTENTS:
Simulation model
...
Identification code
...
MPC code
...
Time domain simulation code
