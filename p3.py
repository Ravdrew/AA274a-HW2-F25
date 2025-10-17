def find_closest_nominal_state(current_state):
    ################ Your code here ############################################
    # Hint: This shouldn't take more than a couple lines
    closest_state_idx = np.argmin(np.linalg.norm(nominal_states - current_state, axis=1))
    ############################################################################
    return closest_state_idx

for i in range(len(nominal_states)):
    #################### Your code here ##########################################
    # Hints:
    # This problem very closely follows the lecture notes! We highly recommend
    # going through them before attempting the problem if you haven't already
    # done so
    # 1. Use planar_quad.get_continuous_jacobians() to calculate the jacobians of the dynamics
    # 2. Use the import ricatti_solver function to get P. Note that this function
    # actually returns the transpose of P
    # 3. Find the gains and update the gains lookup dictionary with it
    # 4. Nominal controls are not defined for the last state. Set these to zero. 
    A, B = planar_quad.get_continuous_jacobians(nominal_states[i], [0,0] if i == len(nominal_states) - 1 else nominal_controls[i])
    P_transposed = ricatti_solver(A, B, Q, R)
    P = P_transposed.T
    R_inverse = np.linalg.inv(R)
    K = R_inverse @ B.T @ P
    ##############################################################################
    gains_lookup[i] = K

def simulate_closed_loop(initial_state, nominal_controls):
    states = [initial_state]
    for k in range(N):
        #################### Your code here ####################################
        # Add code to compute the new controls using the LQR controller
        # Hints:
        # 1. Find the closest nominal state to the current state and lookup
        # the corresponding gain matrix
        # 2. Use the closest nominal state, its corresponding control, and the
        # gain matrix to compute the adjusted controls for the current state'
        cur_state = states[k]
        closest_nominal_state_idx = find_closest_nominal_state(cur_state)
        control = nominal_controls[closest_nominal_state_idx] - gains_lookup[closest_nominal_state_idx] @ (cur_state - nominal_states[closest_nominal_state_idx])
        #######################################################################
        control = np.clip(control, planar_quad.min_thrust_per_prop, planar_quad.max_thrust_per_prop)
        next_state = planar_quad.discrete_step(states[k], control, dt)
        next_state = apply_wind_disturbance(next_state, dt)
        states.append(next_state)
    return np.array(states)