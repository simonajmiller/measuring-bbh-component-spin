# `measuring-bbh-component-spin`: Repository for *Gravitational wave signals carry information beyond effective spin parameters* (Miller et. al. 2023)

This repository contains all the code to reproduce the results in *Gravitational wave signals carry information beyond effective spin parameters* [insert arxiv link]. None of the specific data used is pushed to the repo because the files are large but we give step by step instructions on how to recreate all data used below. The repo is set up with all the requisite folders / organization. Folders that our scripts write data to currently just contain `.gitkeep` files as placeholders. 

## 1. Generate Simulated Population Parameters 

### Organization

- Scripts: `Code/GeneratePopulations/`
- Outputs saved: `Data/InjectedPopulationParameters`

### Instructions to reproduce

The first step to generating mock catalogs of gravitational-wave events is to generate the parameters for each BBH in each population.
First, to generate `.json` files containing the underlying distributions for each of the three populations, run 
```
$ python generate_underlying_pops.py
``` 
These underlying populations are plotted in Figure 1. 
To then generate 50,0000 *found* injections for each population, i.e. those from the underlying distributions that pass a network signal-to-noise-ratio  (SNR) threshold of 10, run 
```
$ python generate_pops.py
``` 
Note that this will take hours to run since most randomly generated parameter combinations to not produce signals that pass the SNR cut.

In `Data/InjectedPopulationParameters` there is a jupyter notebook (`view_populations.ipynb`) to look at the underlying versus detected population parameters for each of the three populations.

Finally, to generate the `.json` file with the sensitivity injections from a flat distribution, which is needed for the selection effects term in population inference (see section 3), run 
```
$ python generate_flat_pop_for_injDict.py
```

## 2. Perform Individual Event Inference 

### Organization
- Scripts: `Code/IndivdualInference/`
- Inputs read from: `Data/InjectedPopulationParameters`
- Outputs saved: `Data/IndividualInferenceOutput` and `Data/PopulationInferenceInput`

### Instructions to reproduce

Next, from the 50,000 events we generated from each population, we want to choose a much smaller subset of events that we will inject into LIGO data. These will be our "catalogs" analogous to the actual events LIGO has detected. In the `makeDagFiles.py` and `launchBilby.py` scripts, we select a subset of the 50,000 found events, inject them Gaussian noise realiziations using O3 actual noise PSDs from LIGO Livingston, LIGO Hanford, and Virgo, and use `bilby` to perform parameter estimation on the signals. 

Run
```
$ python makeDagFiles.py
```
to make a text file containing 300 ID numbers of the randomly selected events for each population, and corresponding `.dag` file in the `condor` sub-folder. These files are used to submit all 300 jobs per population to run `bilby` with `HTCondor` on the LIGO computing cluster.
Also, in the `condor` sub-folder are the necessary `.sub` files that submit the `launchBilby.py` script.

*IMPORTANT NOTE* In line 12 of `makeDagFiles.py`, line 18 of `launchBilby.py`, line 12 of `launchBilby.sh`, and lines 2 8 9 & 10 of each `.sub` file in the `condor` folder you will need to change the repository root to be your own.

Once you have this set up, you are ready to submit to condor. To do this, simply run 
```
$ condor_submit_dag bilby_population1_highSpinPrecessing.dag
``` 
for population 1 (and `bilby_population2_mediumSpin.dag` and `bilby_population3_lowSpinAligned.dag` analogously for the others).

Individual event parameter estimation will take days to weeks to run. 

Once jobs have finished, turn the `bilby` outputs into the correct format to be read into to population inference by running 
```
$ python make_sampleDicts.py
```
Finally, to format the sensitivity injections correctly, run 
```
$ python make_injectionDict_flat.py
```

## 3. Perform Population Level Inference

### Organization
- Scripts: `Code/PopulationInference/`
- Inputs read from: `Data/PopulationInferenceInput`
- Outputs saved: `Data/PopulationInferenceOutput`

### Instructions to reproduce

The final step to reproduce our results is to run population inference using `emcee` on our mock population outputs from `bilby` to see if we can recover the original populations we injected. 
To run the beta+doubleGaussian model (Section III of the paper), run 
```
$ python run_beta_plus_doublegaussian.py POP_NUMBER N_EVENTS N_STEPS
```
where you pass `POP_NUMBER`, `N_EVENTS`, and `N_STEPS` via commandline. `POP_NUMBER` should be "1", "2", or "3". `N_EVENTS` is the number of events you want to run on. In the paper we choose "70" and "300". If you want more than 300 make sure to generate enough events in `makeDagFiles.py` by editing the code to select a larger number. `N_STEPS` is the number of steps we want the `emcee` MCMC sampler to run for. We select 30,000. This usually takes a few days to run.

This is repeated for `run_beta_plus_gaussian.py` to produce the results for Section IV of the paper. 

To look at the `emcee` outputs, use notebooks `inspect_betaPlusDoubleGaussian.ipynb` and `inspect_betaGaussian.ipynb` in the `Data/PopulationInferenceOutput` folder. 

Code to generate all figures is found in the `Figures` folder, which largely takes data from the `Data/PopulationInferenceOutput` as inputs.
