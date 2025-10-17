s_0 = np.array([EGO_START_POS[0], EGO_START_POS[1], np.pi/2])  # Initial state.
s_f = np.array([EGO_FINAL_GOAL_POS[0], EGO_FINAL_GOAL_POS[1], np.pi/2])  # Final state.
def optimize_trajectory(
    time_weight: float = 1.0,
    verbose: bool = True
):
    """Computes the optimal trajectory as a function of `time_weight`.

    Args:
        time_weight: \alpha in the HW writeup.

    Returns:
        t_f_opt: Final time, a scalar.
        s_opt: States, an array of shape (N + 1, s_dim).
        u_opt: Controls, an array of shape (N, u_dim).
    """

    def cost(z):
        ############################## Code starts here ##############################
        # TODO: Define a cost function here
        # HINT: you may find `unpack_decision_variables` useful here. z is the packed 1D representation of t,s and u. Return the value of the cost.

        t, s, u = unpack_decision_variables(z)
        time_step = t / N
        return np.sum((time_weight + u[:, 0] ** 2 + u[:, 1] ** 2) * time_step)
        ############################## Code ends here ##############################

    # Initialize the trajectory with a straight line
    z_guess = pack_decision_variables(
        20, s_0 + np.linspace(0, 1, N + 1)[:, np.newaxis] * (s_f - s_0),
        np.ones(N * u_dim))

    # Minimum and Maximum bounds on states and controls
    # This is because we would want to include safety limits
    # for omega (steering) and velocity (speed limit)
    bounds = Bounds(
        pack_decision_variables(
            0., -np.inf * np.ones((N + 1, s_dim)),
            np.array([0.01, -om_max]) * np.ones((N, u_dim))),
        pack_decision_variables(
            np.inf, np.inf * np.ones((N + 1, s_dim)),
            np.array([v_max, om_max]) * np.ones((N, u_dim)))
    )

    # Define the equality constraints
    def eq_constraints(z):
        t_f, s, u = unpack_decision_variables(z)
        dt = t_f / N
        constraint_list = []
        for i in range(N):
            V, om = u[i]
            x, y, th = s[i]
            ############################## Code starts here ##############################
            # TODO: Append to `constraint_list` with dynanics constraints
            constraint_list.append(s[i + 1] - s[i] - dt * np.array([V * np.cos(th), V * np.sin(th), om]))
            ############################## Code ends here ##############################

        ############################## Code starts here ##############################
        # TODO: Append to `constraint_list` with initial and final state constraints
        constraint_list.append(s[0] - s_0)
        constraint_list.append(s[-1] - s_f)
        ############################## Code ends here ##############################
        return np.concatenate(constraint_list)

    # Define the inequality constraints
    def ineq_constraints(z):
      t_f, s, u = unpack_decision_variables(z)
      dt = t_f / N
      constraint_list = []
      for i in range(N):
          V, om = u[i]
          x, y, th = s[i]
          ############################## Code starts here ##############################
          # TODO: Append to `constraint_list` with collision avoidance constraint
          dist_to_obstacle = np.sqrt((x - OBSTACLE_POS[0])**2 + (y - OBSTACLE_POS[1])**2)
          constraint_list.append(dist_to_obstacle - (EGO_RADIUS + OBS_RADIUS))
          ############################## Code ends here ################################
      return np.array(constraint_list)

    result = minimize(cost,
                      z_guess,
                      bounds=bounds,
                      constraints=[{
                          'type': 'eq',
                          'fun': eq_constraints
                      },
                      {
                          'type': 'ineq',
                          'fun': ineq_constraints
                      }])
    if verbose:
        print(result)

    return unpack_decision_variables(result.x)