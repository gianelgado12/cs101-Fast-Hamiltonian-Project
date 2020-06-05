namespace Operations {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Simulation;
    open Microsoft.Quantum.Oracles;
    open Microsoft.Quantum.Characterization;
    open Microsoft.Quantum.Convert;

    // Terms and coefficients from "Scalable Quantum Simulation of Molecular Energies,"
    // O'Malley et. al. https://arxiv.org/abs/1512.06860.

    // H ≔ a II + b₀ ZI + b₁ IZ + b₂ ZZ + b₃ YY + b₄ XX

    operation sim_ham(ham_idx_strength: (Int, Double)[], step_int : Int , sim_time : Double) : Unit{
        let H2Terms =[[PauliI, PauliI], [PauliZ, PauliI], [PauliI, PauliZ], [PauliZ, PauliZ], [PauliY, 
        PauliY], [PauliX, PauliX]];
        let num_runs = 10;
        using(qs = Qubit[2]){
            for (j in 0..num_runs){
                let gate_length = Length(ham_idx_strength);
                for(i in 0..gate_length-1){
                    let (gate_idx, gate_str) = ham_idx_strength[i];
                    Exp(H2Terms[gate_idx], sim_time* gate_str/IntAsDouble(step_int), qs);
                }
                ResetAll(qs);
            }
        }
        
    }  
    
    


    


}
