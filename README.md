# Curve-Fitting---Back-Propagation-Algorithm
Implementation of Curve Fitting using Neural Network Back Propagation Algorithm

Picked n = 300 real numbers uniformly at random ranged from 0 to 1, as x1, ... , xn.
Picked n = 300 real numbers uniformly at random ranging from -1/10 to 1/10 as v1, ... , vn.
Considered a function di = sin(20xi) + 3xi + vi; i = 1; : : : ; n for which curve fitting needs to be done. 
Plotted the points (xi, di); i = 1, ... ,n.

Used the Backpropagation algorithm with online learning to find the optimal weights/network that minimize
the mean-squared error (MSE).

Plotted the final curve fitting points in the same graph with different color.
Plotted Epoch vs Mean Square Error in another graph
