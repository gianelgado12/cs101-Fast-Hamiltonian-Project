import qsharp
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
from Operations import sim_ham

# Terms and coefficients from "Scalable Quantum Simulation of Molecular Energies,"
# O'Malley et. al. https://arxiv.org/abs/1512.06860.

# H ≔ a II + b₀ ZI + b₁ IZ + b₂ ZZ + b₃ YY + b₄ XX

# Hamiltonian Decomposition Coefficients for Various Bond Lengths of H2
H2Coeff = [
            [0.5678, -1.4508, 0.6799, 0.0791, 0.0791],
            [0.5449, -1.287, 0.6719, 0.0798, 0.0798],
            [0.5215, -1.1458, 0.6631, 0.0806, 0.0806],
            [0.4982, -1.0226, 0.6537, 0.0815, 0.0815],
            [0.4754, -0.9145, 0.6438, 0.0825, 0.0825],
            [0.4534, -0.8194, 0.6336, 0.0835, 0.0835],
            [0.4325, -0.7355, 0.6233, 0.0846, 0.0846],
            [0.4125, -0.6612, 0.6129, 0.0858, 0.0858],
            [0.3937, -0.595, 0.6025, 0.087, 0.087],
            [0.376, -0.5358, 0.5921, 0.0883, 0.0883],
            [0.3593, -0.4826, 0.5818, 0.0896, 0.0896],
            [0.3435, -0.4347, 0.5716, 0.091, 0.091],
            [0.3288, -0.3915, 0.5616, 0.0925, 0.0925],
            [0.3149, -0.3523, 0.5518, 0.0939, 0.0939],
            [0.3018, -0.3168, 0.5421, 0.0954, 0.0954],
            [0.2895, -0.2845, 0.5327, 0.097, 0.097],
            [0.2779, -0.255, 0.5235, 0.0986, 0.0986],
            [0.2669, -0.2282, 0.5146, 0.1002, 0.1002],
            [0.2565, -0.2036, 0.5059, 0.1018, 0.1018],
            [0.2467, -0.181, 0.4974, 0.1034, 0.1034],
            [0.2374, -0.1603, 0.4892, 0.105, 0.105],
            [0.2286, -0.1413, 0.4812, 0.1067, 0.1067],
            [0.2203, -0.1238, 0.4735, 0.1083, 0.1083],
            [0.2123, -0.1077, 0.466, 0.11, 0.11],
            [0.2048, -0.0929, 0.4588, 0.1116, 0.1116],
            [0.1976, -0.0792, 0.4518, 0.1133, 0.1133],
            [0.1908, -0.0666, 0.4451, 0.1149, 0.1149],
            [0.1843, -0.0549, 0.4386, 0.1165, 0.1165],
            [0.1782, -0.0442, 0.4323, 0.1181, 0.1181],
            [0.1723, -0.0342, 0.4262, 0.1196, 0.1196],
            [0.1667, -0.0251, 0.4204, 0.1211, 0.1211],
            [0.1615, -0.0166, 0.4148, 0.1226, 0.1226],
            [0.1565, -0.0088, 0.4094, 0.1241, 0.1241],
            [0.1517, -0.0015, 0.4042, 0.1256, 0.1256],
            [0.1472, 0.0052, 0.3992, 0.127, 0.127],
            [0.143, 0.0114, 0.3944, 0.1284, 0.1284],
            [0.139, 0.0171, 0.3898, 0.1297, 0.1297],
            [0.1352, 0.0223, 0.3853, 0.131, 0.131],
            [0.1316, 0.0272, 0.3811, 0.1323, 0.1323],
            [0.1282, 0.0317, 0.3769, 0.1335, 0.1335],
            [0.1251, 0.0359, 0.373, 0.1347, 0.1347],
            [0.1221, 0.0397, 0.3692, 0.1359, 0.1359],
            [0.1193, 0.0432, 0.3655, 0.137, 0.137],
            [0.1167, 0.0465, 0.362, 0.1381, 0.1381],
            [0.1142, 0.0495, 0.3586, 0.1392, 0.1392],
            [0.1119, 0.0523, 0.3553, 0.1402, 0.1402],
            [0.1098, 0.0549, 0.3521, 0.1412, 0.1412],
            [0.1078, 0.0572, 0.3491, 0.1422, 0.1422],
            [0.1059, 0.0594, 0.3461, 0.1432, 0.1432],
            [0.1042, 0.0614, 0.3433, 0.1441, 0.1441],
            [0.1026, 0.0632, 0.3406, 0.145, 0.145],
            [0.1011, 0.0649, 0.3379, 0.1458, 0.1458],
            [0.0997, 0.0665, 0.3354, 0.1467, 0.1467],
            [0.0984, 0.0679, 0.3329, 0.1475, 0.1475]
        ]

# Coefficients for the Identity Trandformation portion of the Hamiltonian Decomposition
H2IdentityCoeff = [
            2.8489, 2.1868, 1.7252, 1.3827, 1.1182, 0.9083, 0.7381, 0.5979, 0.4808, 0.3819, 0.2976, 0.2252, 0.1626, 0.1083, 0.0609, 0.0193, -0.0172, -0.0493, -0.0778, -0.1029, -0.1253, -0.1452, -0.1629, -0.1786, -0.1927, -0.2053, -0.2165, -0.2265, -0.2355, -0.2436, -0.2508, -0.2573, -0.2632, -0.2684, -0.2731, -0.2774, -0.2812, -0.2847, -0.2879, -0.2908, -0.2934, -0.2958, -0.298, -0.3, -0.3018, -0.3035, -0.3051, -0.3066, -0.3079, -0.3092, -0.3104, -0.3115, -0.3125, -0.3135
        ]

# 2-Local gates that make up the complete hamiltonian (0 = Identity, 1 = Pauli X, 2 = Pauli Y, 3 = Pauli Z)
H2Terms = [
            [0, 0], [3, 0], [0, 3], [3, 3], [2, 2], [1, 1]
        ]

# Implementation of qDRIFT protocal to contruct simulation circuit from Campbell paper        
def qDrift(bond_ind, sim_time, e_prec, num_terms = 6):
    H_coeffs = H2Coeff[bond_ind].copy()
    H_coeffs.insert(0, H2IdentityCoeff[bond_ind])
    H_coeffs = H_coeffs[0:num_terms]
    coeff_sum = np.sum(np.abs(H_coeffs))

    N = np.ceil((2 * (coeff_sum ** 2) * (sim_time**2))/e_prec)
    V = []
    H_probs = np.abs(np.array(H_coeffs)/coeff_sum)
    for _ in range(int(N)):
        hamIdx = np.random.choice(np.arange(0, num_terms), p = H_probs)
        V.append((int(hamIdx), coeff_sum))
    return (V, int(N))

# Implementation of First-Order Trotter-Suzuki compilation of simulation circuit
def first_order_trot_suzuki(bond_ind, sim_time, e_prec, num_terms = 6):
    H_coeffs = H2Coeff[bond_ind].copy()
    H_coeffs.insert(0, H2IdentityCoeff[bond_ind])
    H_coeffs = H_coeffs[:num_terms]

    m_lambda = np.max(np.abs(H_coeffs))
    r = np.ceil((num_terms**2 * m_lambda**2 * sim_time**2)/(2*e_prec))
    r = int(r)
    V = []
    for _ in range(r):
        for i in range(num_terms):
            V.append((i, H_coeffs[i]))
    return (V, r)

# Simulation Parameters
sim_time = 1
bond_idx = 2

# Lists to hold simulation data
log_e_dom = []
trot_times = []
rand_times = []

# Running Simulation with both Trotter and qDrift for various levels of precision
for i in np.arange(0, 2, .05):
    e_prec = 10.0**(-1*i)
    rand_start = time.time()
    V, r = qDrift(bond_idx, sim_time, e_prec)
    sim_ham.simulate(ham_idx_strength = V, step_int = r, sim_time = 1.0)
    rand_end = time.time()
    trot_start = time.time()
    V, r = first_order_trot_suzuki(bond_idx, sim_time, e_prec)
    sim_ham.simulate(ham_idx_strength = V, step_int = r, sim_time = 1.0)
    trot_end = time.time()
    log_e_dom.append(i)
    trot_times.append(trot_end-trot_start)
    rand_times.append(rand_end-rand_start)

plt.plot(log_e_dom, rand_times, label='qDrift')
plt.plot(log_e_dom, trot_times, label='First Order Trotter')
plt.xlabel("-Log10 Precision")
plt.ylabel("Run Time")
plt.legend()
plt.savefig("e_precision_comp.png")
plt.show()


num_terms_dom = [1,2,3,4,5,6]
trot_times = []
rand_times = []

# Running Simulation with both Trotter and qDrift for various numbers of terms in Hamiltonian
for i in num_terms_dom:
    e_prec = 0.1
    rand_start = time.time()
    V, r = qDrift(bond_idx, sim_time, e_prec, num_terms=i)
    sim_ham.simulate(ham_idx_strength = V, step_int = r, sim_time = 1.0)
    rand_end = time.time()
    trot_start = time.time()
    V, r = first_order_trot_suzuki(bond_idx, sim_time, e_prec, num_terms=i)
    sim_ham.simulate(ham_idx_strength = V, step_int = r, sim_time = 1.0)
    trot_end = time.time()
    trot_times.append(trot_end-trot_start)
    rand_times.append(rand_end-rand_start)

plt.plot(num_terms_dom, rand_times, label='qDrift')
plt.plot(num_terms_dom, trot_times, label='First Order Trotter')
plt.xlabel("Number of Terms in Hamiltonian")
plt.ylabel("Run Time")
plt.legend()
plt.savefig("num_terms_comp.png")
plt.show()