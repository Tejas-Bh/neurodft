# neurodft

a unique spiking neural network (SNN) written in `numpy`.

---

## How it works
This network consists of a rectangular grid of nodes ("neurons").

Each node has a membrane potential. The model uses "leaky-integrate and fire", a method used to model action potentials in neurons to update and adjust the membrane voltages of each node.

Each node has a resonant frequency. These frequencies are arranged via a gradient from "top" to "bottom" of the grid.

When a signal is passed into the network, nodes accumulate membrane voltage and fire based on how much of the node's resosnant frequency is in the original signal.

This allows for the **spectral decomposition** of input signals, similar to a Fourier Transform (hence the name, neuro[morphic] discrete Fourier Transform).
