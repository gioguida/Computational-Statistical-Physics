# Particle Track Reconstruction as a Spin-Glass Optimization Problem

## The Idea

Track reconstruction in high-energy physics, such as at CERN, is a large combinatorial problem. This project proposes mapping a simplified 2D version of the task onto an Ising-like spin-glass model. Potential track segments between detector layers are treated as binary variables (spins). By designing a Hamiltonian that rewards smooth trajectories and strongly penalizes bifurcations, track finding becomes an energy minimization problem.

## Project Scope and Division of Labor

- Generate a 2D toy dataset of particle hits with added noise across concentric layers in Python.
- Construct the interaction matrix ($J_{ij}$) based on the geometry of connecting segments in C++.
- Implement a simulated annealing / Metropolis-Hastings algorithm to cool the system and find the ground state corresponding to the true particle tracks in C++.
- Visualize the reconstructed tracks and evaluate the algorithm's accuracy in Python.
