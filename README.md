# neurodft

The first model applying RF coupling to neuromorphic AI.

This is a `numpy` implementation of a neural compute model based on electromagnetic coupling.

The network uses a **gradient-frequency neural network** (Large 2010), in that different portions of the network are resonant to different spectral portions/frequencies of the input signal, similar to the mammalian auditory cortex.

This allows spectral decomposition to be an **intrinsic phyiscal property** of the network *(hence the name, neuromorphic-developed-fourier-transform, or neurodft)*.

I'm currently working on writing a formal specification for hardware that implements this architecture via WiNoC RF.

Feel free to reach out at `tejas dot bhagawatula at gmail dot com`!
