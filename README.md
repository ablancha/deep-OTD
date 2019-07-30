# deep-OTD

Source code for data-driven learning of OTD modes used in "Machine Learning the Tangent Space of Dynamical Instabilities from Data" (https://arxiv.org/abs/1907.10413)

The optimally time-dependent (OTD) modes are a set of deformable orthonormal tangent vectors that track directions of instabilities along any trajectory of a dynamical system. Traditionally, these modes are computed by a time-marching approach that involves solving multiple initial-boundary-value problems concurrently with the state equations. However, for a large class of dynamical systems, the OTD modes are known to depend “pointwise” on the state of the system on the attractor, and not on the history of the trajectory. We leverage the power of neural networks to learn this “pointwise” mapping from phase space to OTD space directly from data. The result of the learning process is a cartography of directions associated with strongest instabilities in phase space, as well as accurate estimates for the leading Lyapunov exponents.

Send comments and questions to ablancha@mit.edu
