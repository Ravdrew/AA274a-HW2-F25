class DoubleIntegratorDynamics(Dynamics):

    def __init__(self) -> None:
        super().__init__()
        self.xdd_max = 0.5 # m/s^2
        self.ydd_max = 0.5 # m/s^2
        self.n = 4
        self.m = 2

    def feed_forward(self, state:np.array, control:np.array):

        num_rollouts = int(state.shape[0] / self.n)

        # Define Gaussian Noise
        if self.noisy == False:
            var = np.array([0.0, 0.0, 0.0, 0.0])
        elif self.noisy == True:
            var = np.array([0.01, 0.01, 0.001, 0.001])

        var_stack = np.tile(var, (num_rollouts))
        w = np.random.normal(loc=np.zeros(state.shape), scale=var_stack)

        # State space dynamics
        A = np.array([[1.0, 0.0, self.dt, 0.0],
                    [0.0, 1.0, 0.0, self.dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])

        B = np.array([[0.0, 0.0],
                    [0.0, 0.0],
                    [self.dt, 0.0],
                    [0.0, self.dt]])

        # Stack to parallelize trajectories
        A_stack = np.kron(np.eye(num_rollouts), A)
        B_stack = np.tile(B, (num_rollouts, 1))

        ############################## Code starts here ##############################
        """
        Construct "state_new" by applying discrete time dynamics vectorized equations. Will require use of "A_stack" and "B_stack".
        """
        velocities = B_stack @ control
        prev = A_stack @ state
        state_new = prev + velocities

        ############################## Code ends here ##############################

        # Add noise
        state_new = state_new + w

        return state_new

    def rollout(self, state_init, control_traj, num_rollouts):

        num_steps = control_traj.shape[1]

        state_traj = np.zeros((self.n*num_rollouts, num_steps+1))
        state_traj[:,0] = np.tile(state_init, num_rollouts)
        ############################## Code starts here ##############################
        """
        Populate "state_traj" using only one for-loop, along with the "feed_forward" function above.
        """
        for step in range(1, num_steps+1):
            state_traj[:,step] = self.feed_forward(state_traj[:, step-1], control_traj[:, step-1])
        ############################## Code ends here ##############################

        return state_traj
