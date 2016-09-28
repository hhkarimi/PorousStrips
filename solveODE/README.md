## odeFD.py
Solves integro-differential equation desciribing plate bending (Kirchhoff equations) loaded by viscous fluid drag of a presumed form with a finite-difference approach.  Since the system is nonlinear, due to large deformations, an iterative algorithm with a weighted average is applied.  Convgergence time on a 2015 Macbook is O(0.2 seconds).

The numerical algorithm in odeFD.py forms the core of the predictive model in this project, refined by optimization (../optimizeNumerics).
