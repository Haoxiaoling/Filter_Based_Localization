import numpy as np

from utils import minimized_angle

class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE
        particles = np.zeros((self.num_particles, 3))
        weights = np.ones(self.num_particles)
        for i in range(self.num_particles):
            sigma = env.V(self.particles[i,:], u) @env.noise_from_motion(u, self.alphas) @ env.V(self.particles[i,:], u).T
            particles[i,:] = np.random.multivariate_normal(env.forward(self.particles[i,:], u).ravel(),  sigma)
            particles[i,2] = minimized_angle(particles[i,2])
            dz = np.array(minimized_angle(env.observe(particles[i,:].ravel(), marker_id) - z)).reshape(-1, 1)
            weights[i] = env.likelihood(dz, self.beta)# * self.weights[i]
        weights = weights/np.sum(weights)
        self.particles, self.weights = self.resample(particles, weights)
        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        # new_particles, new_weights = particles, weights	
        new_particles = np.zeros((self.num_particles, 3))
        new_weights = np.ones(self.num_particles)
        # YOUR IMPLEMENTATION HERE
        r = (1.0/self.num_particles) * np.random.rand()
        c = weights[0]
        i = 0
        for m in range(self.num_particles):
            U = r + float(m)/self.num_particles
            while U > c:
                i = i + 1
                c = c + weights[i]
            new_particles[m, :] = particles[i, :]
            new_weights[m] = weights[i]
        # new_weights = new_weights/np.sum(new_weights)
        return new_particles, new_weights

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.cos(particles[:, 2]).sum(),
            np.sin(particles[:, 2]).sum()
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles

        return mean.reshape((-1, 1)), cov
