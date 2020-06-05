namespace Operations {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Simulation;
    open Microsoft.Quantum.Oracles;
    open Microsoft.Quantum.Characterization;
    open Microsoft.Quantum.Convert;

    // Operation that simulates hamiltonian evolution over a given time with prescribed time steps
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
