# CS101-Fast-Hamiltonian-Project

This project aims to analyze the complexity of two methods of hamiltonian compilation: First Order Trotter Suzuki and qDRFIT

To run the various tests invoke the host file `host.py` from the command line with the following arguments:
* `0` - To run precision comparison test
* `1` - To run term number comparison test
* `2` - To run simulation time comparison test

Each test will output two plots. The first plot will be comparing how the run time scales for the given metric and the second plot will compare how the number of gates scales for this same metric.