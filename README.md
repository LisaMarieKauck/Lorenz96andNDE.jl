# Lorenz96andNDE

[![Build Status](https://github.com/LisaMarieKauck/Lorenz96andNDE.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/LisaMarieKauck/Lorenz96andNDE.jl/actions/workflows/CI.yml?query=branch%3Amaster)

This package creates an ODE Problem with a Lorenz96 system. As a solver for the problems the Tsitouras 5/4 Runge-Kutta method is used.The
solved data gets split into two sets of training and validation data, respectively. For more clarity only data starting from a certain threshold will be used. A neural network is added to the Lorenz96 system which defines the neural ODE. It will then be trained according to the minimization of a loss function. For that an Adam optimizer is used which is a form of a stochastic gradient descent.