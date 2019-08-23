# deep-OTD

Source code for data-driven learning of OTD modes used in "Machine Learning the Tangent Space of Dynamical Instabilities from Data" by Blanchard \& Sapsis (https://arxiv.org/abs/1907.10413)

## Background 

The optimally time-dependent (OTD) modes are a set of deformable orthonormal tangent vectors that track directions of instabilities along any trajectory of a dynamical system. Traditionally, these modes are computed by a time-marching approach that involves solving multiple initial-boundary-value problems concurrently with the state equations. However, for a large class of dynamical systems, the OTD modes are known to depend “pointwise” on the state of the system on the attractor, and not on the history of the trajectory. We leverage the power of neural networks to learn this “pointwise” mapping from phase space to OTD space directly from data. The result of the learning process is a cartography of directions associated with strongest instabilities in phase space, as well as accurate estimates for the leading Lyapunov exponents.

## File description

* [src/](src/) contains source code for data generation and dOTD learning
* [examples/3dim/](examples/3dim/) contains files for the three-dimensional nonlinear system (run with main.py, plot with plt\*.m)
* [examples/cdv/](examples/cdv/) contains files for the Charney-DeVore system (run with main.py, plot with plt\*.m)


## Dependencies

* [autograd](https://github.com/HIPS/autograd)
* [numpy](http://www.numpy.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)
* [MATLAB](https://www.mathworks.com) (only for reproducing figures)

Send comments and questions to ablancha@mit.edu
