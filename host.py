import qsharp
import numpy as np
import random
import math

from Operations import H2Coeff, H2IdentityCoeff, H2Terms

def qDrift(bond_ind, sim_time, e_prec):
    H_coeffs = H2Coeff.simulate(idxBondLength = bond_ind)
    H_coeffs.insert(0, H2IdentityCoeff.simulate(idxBond = bond_ind))
    
    coeff_sum = np.sum(np.abs(H_coeffs))

    N = np.ceil((2 * (coeff_sum ** 2) * (sim_time**2))/e_prec)

    V = []
    H_probs = np.abs(np.array(H_coeffs)/coeff_sum)
    print(N)
    for _ in range(int(N)):
        hamIdx = np.random.choice(np.arange(0, len(H_coeffs)), p = H_probs)
        V.append(H2Terms.simulate(idxHamiltonian = int(hamIdx)))
    return (V, N)

print(qDrift(2, 30, 0.1))

