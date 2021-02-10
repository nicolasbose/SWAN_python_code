# SWAN-code

This project aim to analyse the sensibilbity of SWAN wave model in representing the swell and windsea condition for the souther Brazil nearshore, when using different database as input.

SWAN wave modelo version 41.31 (http://swanmodel.sourceforge.net/)

Python SWAN code: Generate wind and wave boundary condition, generate 2D-spectrum from waves partition and data analysis

It must be open in Jupyter lab! 

This repository provides the following code in python:

Generate wind and wave boundary condition:
  - Wind from ERA5, GFS and CSIRO.
  - Wave from ERA5, WW3/NCEP and CSIRO
 
 Generate 2D-spectrum from waves partition:
  - Waves partition from CSIRO database are used to build directinal-wave-spectrum, based on JONSWAP waves formulation.
  
 Data analysis:
  - Statistical anaysis for SWAN wave model results
