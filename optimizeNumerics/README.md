# optimizeNumerics

## `optimC.py`
Loads indicated matlab data file (here, `U0.9L59.14.mat`) defining the bending profile of an elastic strip as it is dragged through viscous fluid.  Fluid-solid drag model is fitted to the experimental curve by optimzing the two drag coefficients in the normal and tangential direction (`cn` and `ct`).  These coefficients are found to be constant over a range of parameters, providing a predictive model of the strip bending given system parameters.
