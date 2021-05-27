<h2>Growing Neural Cellular Automata</h2>

Reconstruction of [1] in pytorch with CNN model, as suggested by the author in [2].
Model is capable of growing and preserving state.

<h3>Train and test</h3>

Clone repository and run [main.py](https://github.com/Sergo2020/Neural_Automata_pytorch/blob/master/main.py). Hyper parameters are pre-set in the same file in "hyper" dictionary,
therefore it is intended for use with python IDE. 

By default training procedure will start followed up by test.

For testing pre-trained model, just run test function with default settings. The output of the test procedure is a *.gif file with animated growth as sequences below:

<p align="center">
<img width="64" height="64" src="https://github.com/Sergo2020/Neural_Automata_pytorch/blob/master/results/output_gif.gif">
</p>
<h3>References</h3>

[[1]](https://distill.pub/2020/growing-ca/) Mordvintsev, et al., "Growing Neural Cellular Automata", Distill, 2020.  
[[2]](https://www.youtube.com/watch?v=kA7_LGjen7o&t=1095s&ab_channel=ODSAIGlobalODSAIGlobal) Mordvintsev, A. (2020, October 9). Alexander Mordvintsev: neural cellular automata from scratch. Youtube. 
