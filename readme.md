<h2>Growing Neural Cellular Automata</h2>

>Imagine if we could design systems of the same plasticity and robustness as biological life:
> structures and machines that could grow and repair themselves. 
> Such technology would transform the current efforts in regenerative medicine, 
> where scientists and clinicians seek to discover the inputs or stimuli that could cause cells in the body to build structures on demand as needed. 
> To help crack the puzzle of the morphogenetic code, and also exploit the insights of biology to create self-repairing systems in real life,
> we try to replicate some of the desired properties in an in silico experiment.

Reconstruction of Growing Neural Cellular Automata[1] in pytorch with CNN model, as suggested by the author [2].
Model is capable of growing and preserving state.

<h3>Train and test</h3>

Clone repository and run [main.py](https://github.com/Sergo2020/Neural_Automata_pytorch/blob/master/main.py). Hyper parameters are pre-set in the same file in "hyper" dictionary,
therefore it is intended for use with python IDE. 

By default training procedure will start followed up by test.

For testing pre-trained model, just run test function with default settings. The output of the test procedure is a *.gif file with animated growth as sequences below:

<p align="center">
  <img src="https://github.com/Sergo2020/Neural_Automata_pytorch/blob/master/results/ra2_alpha_60_gif.gif" width="100" alt="Red Alert 2" />
  <img src="https://github.com/Sergo2020/Neural_Automata_pytorch/blob/master/results/xcom_fix_60_gif.gif" width="100"  alt="X-COM" />
  <img src="https://github.com/Sergo2020/Neural_Automata_pytorch/blob/master/results/mario_alpha_60_gif.gif" width="100"  alt="Mario Bros." />
</p>

<h3>References</h3>

[[1]](https://distill.pub/2020/growing-ca/) Mordvintsev, et al., "Growing Neural Cellular Automata", Distill, 2020.  
[[2]](https://www.youtube.com/watch?v=kA7_LGjen7o&t=1095s&ab_channel=ODSAIGlobalODSAIGlobal) Mordvintsev, A. (2020, October 9). Alexander Mordvintsev: neural cellular automata from scratch. Youtube. 
