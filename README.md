# EE6227
## Assignment 1


### Setup on which the code was tested
- python==3.6
- numpy==1.19.2 



### Test Functions

<img src='/Imgs/A_F.png'>
<img src='/Imgs/Pasted Graphic.png'>


### PSO

<img src='/Imgs/Func 1 [Sphere function].png'>
<img src='/Imgs/Func7.png'>

### CLPSO

<img src='/Imgs/Func_1 [Sphere function]W0.8.png'>
<img src='/Imgs/unc_5 [Wei.png'>

###  Issues
When using tournament selection to generate exemplars, instead of taking two particles and comparing them, we shuffled the p_best and chose the top %20 as the set for new exemplars, this measure accelerates the process and also has a good performance. 
