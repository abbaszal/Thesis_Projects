# Federated Learning: Stability and Efficiency under Realistic Conditions

Federated Learning (FL) is a decentralized machine learning paradigm that enables multiple clients to collaboratively train a global model without sharing their raw data. While FL offers promising real-world applications, it introduces complexities such as data heterogeneity and varying client participation (e.g., reliable and unreliable clients). 

This study focuses on exploring and optimizing the **stability** and **efficiency** of federated learning systems under realistic conditions using two distinct datasets and **game-theoretic approaches**.

---

## Research Objectives

The primary objective of this research is to:
- Simulate realistic federated learning scenarios.
- Investigate whether an **optimal strategy** exists to achieve:
  - High global model accuracy.
  - Stable and effective client participation.

To address these challenges, this research utilizes concepts from **game theory**, including:
- **Nash Equilibrium**: To analyze the stability of client behaviors.
- **Shapley Values**: To measure and reward individual client contributions.

---

## Methodology

- **Datasets Used**:
  - A spam-based dataset, manually partitioned among 10 clients.
  - An activity recognition dataset, which originally consisted of data collected from 18 clients but i useed 10 clients for simulation.

- **Simulation Setup**:
  - Federated learning simulations were conducted under various configurations to evaluate:
    - Global model accuracy.
    - Client-specific contributions and accuracies.
  - All possible combinations of client interactions were analyzed to balance global accuracy and the stability of individual contributions.

---

## Results

The study provides valuable insights into achieving a **stable** and **efficient** state in federated learning systems. The findings highlight strategies to maximize utility for both the global system and individual clients, emphasizing fairness and stability in collaborative training environments.

---

## Acknowledgment

I am deeply grateful to my supervisor, **Professor Thomas Marchioro**, for his invaluable guidance, advice, and support throughout this research. His expertise and encouragement played a crucial role in the success of this study.
