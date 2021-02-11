# SWAN-code

This project aims to analyze the sensibility of SWAN wave model in representing the swell and windsea condition for the southern Brazil nearshore when using different databases as input.

SWAN wave modelo version 41.31 (http://swanmodel.sourceforge.net/).

Python SWAN code: Generate wind and wave boundary condition, generate 2D-spectrum from waves partition and data analysis

All python scripts must be open in Jupyter lab! 

This repository provides the following code in python:

Generate wind and wave boundary condition:
  - Wind from ERA5, GFS, and CSIRO.
  - Wave from ERA5 (TPAR format), WW3/NCEP (TPAR format), and CSIRO (Spectrum format)
 
 Generate 2D-spectrum from waves partition:
  - Waves partition from CSIRO database are used to build directional-wave-spectrum, based on JONSWAP spectrum formulation.
  
 Data analysis:
  - Statistical analysis for SWAN wave model results
