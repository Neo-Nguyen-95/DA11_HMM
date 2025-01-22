def viterbi_algorithm(psi, pi, A, B, O_index):

    num_state = len(psi)
    
    for t, obs in enumerate(O_index):
        if t == 0:  # Initial observation emission probability
            delta = pi * B[:, obs].reshape(-1, 1)
    
        else:
            # Take N^2 equation
            probability_observation = delta * A * B[:, obs]
            
            for current_index in range(num_state):
                # In each group of calculation, take the max out of N results
                # delta store max value of current_index branch
            
                delta[current_index] = probability_observation[:, current_index].max()
                
                for previous_index in range(num_state):
                    # In each group of calculation, sorted by current_index
                    # Take argmax at current_index as psi
                    # backtracking with previous_index to find previous sequence
                    
                    if probability_observation[:, current_index].argmax() == previous_index:
                        psi[current_index] = psi[previous_index][:t] + [current_index]
                        # only take into account up to time t of the sequence
                        # avoid update the updated psi if 2 route start from a state
    
    for state in range(num_state):
        if delta.argmax() == state:
            sequence = psi[state]
    
    return (delta.max(), sequence)