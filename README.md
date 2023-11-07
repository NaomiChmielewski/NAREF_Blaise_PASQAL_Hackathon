# pasqal_hackathon_official
## Github repository with all relevant notebooks and plots for the hackathon 2023

Notebooks are annotated.

The relevant hyperparameters to reproduce our results can be found in the appendix of our pdf document.

The quantum reservoir can be found in the notebook "QRCvsCRC.ipynb" and is the main body of work.

The notebook "plots.ipynb" is used to create the hyperparameter comparison plots in our pdf document. 

The notebook "extrapolate.ipynb" is used to create the extrapolation plots in our pdf document.

The notebook "CRCvsRNN.ipynb" is used to compare performance an energy consumption between a classical reservoir and an RNN. It is an old notebook without annotations but can be used to refer to some of our claims in the pdf document.7

The notebook "Energy_Consumption_Analysis.ipynb" is used to compare the energy consumption of our quantum algorithm to an RNN on Fresnel and 500 vs. a basic GPU server and the Jolio-Curie Rome.

The folder "sine" contains the results of the hyperparameter searches on the quantum reservoir. Contrary to the name of the folder and files, the files contain the results for both sine and mackey glass.

The folder "plots" contains the plots that were created with the notebooks "plots.ipynb" and "extrapolate.ipynb".


## Packages used

pyRAPL 0.2.3.1
reservoirpy 0.3.8

pyRAPL needs a special access. After installation, write
$ sudo chmod -R a+r /sys/class/powercap/intel-rapl 
in the terminal
