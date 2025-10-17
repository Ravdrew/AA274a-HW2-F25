def feed_forward(self, state:TurtleBotState, control:TurtleBotControl):
    # Define Gaussian Noise
    if self.noisy == False:
      var = np.array([0.0, 0.0, 0.0])
    elif self.noisy == True:
      var = np.array([0.01, 0.01, 0.001])
    w = np.random.normal(loc=np.array([0.0, 0.0, 0.0]), scale=var)


    state_new = TurtleBotState()
    ############################## Code starts here ##############################
    """
    Populate "state_new" by applying discrete time dynamics equations. Use "self.dt" from the Dynamics base class.
    """
    state_new.x = state.x + control.v * np.cos(state.th) * self.dt
    state_new.y = state.y + control.v * np.sin(state.th) * self.dt
    state_new.th = state.th + control.o * self.dt
    ############################## Code ends here ##############################

    # Add noise
    state_new.x  = state_new.x  + w[0]
    state_new.y  = state_new.y  + w[1]
    state_new.th = state_new.th + w[2]
    return state_new

def rollout(self, state_init, control_traj, num_rollouts):
    num_steps = control_traj.shape[1]

    state_traj_rollouts = np.zeros((self.n*num_rollouts, num_steps+1))
    ############################## Code starts here ##############################
    """
    Use two for-loops to loop through "num_rollouts" and "num_steps" to populate "state_traj_rollouts". Use the "feed_forward" function above.
    """

    for rollout in range(num_rollouts):
    state = TurtleBotState(state_init.x, state_init.y, state_init.th)
    state_traj_rollouts[rollout*self.n, 0] = state.x
    state_traj_rollouts[rollout*self.n + 1, 0] = state.y
    state_traj_rollouts[rollout*self.n + 2, 0] = state.th
    for step in range(1, num_steps+1):
        control = TurtleBotControl(v=control_traj[0, step-1], o=control_traj[1, step-1])
        state = self.feed_forward(state, control)
        state_traj_rollouts[rollout*self.n, step] = state.x
        state_traj_rollouts[rollout*self.n + 1, step] = state.y
        state_traj_rollouts[rollout*self.n + 2, step] = state.th
    ############################## Code ends here ##############################

    return state_traj_rollouts