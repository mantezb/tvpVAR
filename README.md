# Stochastic Model Specificaction Search for TVP-VAR-SV

MSc project based on methodology proposed by Eisenstat et al.(2016), applied to TVP-VAR-SV model for monetary policy transmission mechanism analysis.

Code:

* main.py - the main script to transform the data and run MCMC simulation
* diagnostics.py - performs MCMC diagonstic via trace plots and calculation of inefficiency factors
* analytics.py - calculates impulse responses for TVP-VAR-SV and TPV-VECM-SV and plots them
* data_conversion.py - used for analysis of data using time-series plots, histograms and cointegration tests (Johansen). Transforms data using log transformation.

utils:

* coint_johansen.py - Johansen cointegration test by Johansen (1988); ucalled in data_conversion.py
* hpr_sampler.py -  this function samples omega's and the hyperparameters conditional on all other parameters and data; called in main.py
* ineff_factor.py  - functions to calculate inefficiency factors; called in diagonistics.py
* ir_var_sv.py - this function computes impulses responses for the VAR models with stochastic volatility. Structural parameters are recovered from the estimated parameters using lower triangular identification scheme as per Primiceri (2005) recovered from the values simulated by MCMC algorithm by Eisenstat (2016); called in analytics.py
* ir_vecm_sv.py - function to estimate impulse responses for VECM-SV model; used in analytics.py
* mvsvrm.py - function  that simulates log-volatilities for a multivariate stochastic volatility model with independent random-walk transitions; called in main.py
* tnr.py -  univariate truncated normal sampling based on Robert (1995); called in main.py via hpr_sampler.py
* solve_struct.py - this function recovers the structural parameters from the estimated parameters for VECM-SV described in Appendix to Eisenstat et al. (2016); called in analytics.py via ir_vecm_sv.py

plotting:

* plots_diagnostics.py - plots for diagnostics (ineffficiency factors) and time-invariance probabilities across a number of runs
* plots_SMSS_comp.py - plots comparing results for 2 chosen runs 
* plots_SMSS_time.py   - plots to identify time-variation of impulse responses
