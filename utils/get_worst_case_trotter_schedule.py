import numpy as np

def get_worst_case_trotter_schedule(
    qiskit_evs: dict,      # Trotter results: {'k = 1': {'0': [...], ...}, ...}
    exact_evs: dict,       # Exact results: {'0': [...], '1': [...], ...}
    t: list,               # Time array
    k_max: int,            # Maximum available Trotter steps (e.g., 4)
    threshold: float,      # Target percentage error threshold (e.g., 5.0 for 5%)
    measure_qubits: list   # List of qubit keys to monitor (e.g., ['0', '3'])
) -> dict:
    
    # --- 1. SETUP AND INITIALIZATION ---
    
    # Pre-calculate the normalization factor for each qubit (Max value / 100)
    normalization_factors = {
        q_key: np.max(exact_evs[q_key]) / 100.0 
        for q_key in measure_qubits
    }
    
    # Initialize the output schedule structure
    worst_case_schedule = { f'k = {int(i)}' : [] for i in range(1, k_max + 1) }

    # Variables for tracking warnings and remaining time instants
    unmet_time_start_index = -1
    unmet_time_start_value = -1.0
    
    # Sorted list of available Trotter steps
    K_sorted = list(range(1, k_max + 1)) 

    # --- 2. MAIN ADAPTIVE LOOP ---
   
    for i in range(len(t)):
        time_instant = t[i]
        
        # At each time instant, track the MAX required k across all measured qubits
        max_required_k_at_t = 0
        
        # Loop over every measured qubit to find the "worst case" k
        for q_key in measure_qubits:
            q_max_val = normalization_factors[q_key]
            
            # Find the minimum k *required* by this specific qubit at this time
            required_k_for_qubit = k_max + 1 
            
            for k in K_sorted:
                k_key = f'k = {k}'
                
                # Calculate Normalized Percentage Error
                error = np.abs(qiskit_evs[k_key][q_key][i] - exact_evs[q_key][i]) / q_max_val
                
                if error <= threshold:
                    required_k_for_qubit = k
                    break # Found sufficient k for this qubit, move to the next qubit
            
            # Update the max required k for the whole system at this time instant
            max_required_k_at_t = max(max_required_k_at_t, required_k_for_qubit)

        # --- 3. SCHEDULE ASSIGNMENT AND ERROR HANDLING ---
        
        if max_required_k_at_t <= k_max:
            # Case A: Sufficient k was found. Assign time to the required k.
            k_key_final = f'k = {max_required_k_at_t}'
            worst_case_schedule[k_key_final].append(time_instant)
            
            # Reset warning tracker if a sufficient k was found after an unmet period
            unmet_time_start_index = -1 
            unmet_time_start_value = -1.0
            
        else:
            # Case B: The maximum available k (k_max) was insufficient.
            
            # Append the time instant to the k_max schedule for completeness
            worst_case_schedule[f'k = {k_max}'].append(time_instant)
            
            # Start tracking the warning period
            if unmet_time_start_index == -1:
                unmet_time_start_index = i
                unmet_time_start_value = time_instant

    # --- 4. PRINT WARNING (Outside the loop for cleaner output) ---

    if unmet_time_start_index != -1:
        # Note: The warning is printed if the *last* time point was unmet, or if 
        # an unmet period occurred and was not fully closed by the loop.
        print(f"⚠️ WARNING: The target threshold ({threshold}%) was **NOT** met from time t = {unmet_time_start_value}.")
        print(f"These {len(t) - unmet_time_start_index} time instants have been assigned to k = {k_max}, but the true error likely **surpasses** the threshold.")
        print(f"Consider running simulations with k > {k_max} to achieve the desired accuracy for t >= {unmet_time_start_value}.")

    return worst_case_schedule