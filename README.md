# BesselK — Modified Bessel Functions of the Second Kind (PyTorch)

A lightweight, autograd-friendly implementation of the **modified Bessel functions of the second kind**  
\( K_\nu(x) \) for integer orders using **PyTorch**.

## Features

- Implements \( K_0(x) \), \( K_1(x) \), and integer-order \( K_n(x) \)
- Accurate for both **small** and **large** arguments:
  - **Small x:** exact analytical series with harmonic numbers  
  - **Large x:** rational approximation of \( e^{-x}/\sqrt{x} \) form
- Fully differentiable (PyTorch autograd)
- Works on CPU / GPU (`float64` recommended)
