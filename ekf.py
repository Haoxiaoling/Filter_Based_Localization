import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE
        # Update
        # print(env.V(self.mu, u))
        mean = env.forward(self.mu, u)
        sigma = env.G(self.mu, u) @ self.sigma @ env.G(self.mu, u).T + env.V(self.mu, u) @env.noise_from_motion(u, self.alphas) @ env.V(self.mu, u).T
        S = env.H(mean, marker_id).dot(sigma.dot(env.H(mean, marker_id).T)) + self.beta
        K = sigma.dot(env.H(mean, marker_id).T*np.linalg.inv(S))
        self.sigma = (np.identity(3) - K * env.H(mean, marker_id)) @ sigma
        self.mu = mean + K * minimized_angle(z - env.observe(mean, marker_id))
        self.mu[-1] = minimized_angle(self.mu[-1])
        return self.mu, self.sigma
