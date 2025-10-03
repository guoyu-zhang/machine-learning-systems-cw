# Machine Learning Systems - 2024/2025

This course focuses on building and deploying machine learning systems, with hands-on programming tasks, paper writing, and peer reviews.

## Tasks

### Task 1: GPU Programming

In this task, we implemented a custom high-performance KNN operator using Cupy. This task was divided into two parts:

- **Part 1: K-Nearest Neighbors (KNN)**: We implemented four different distance functions (Cosine, L2, Dot Product, and Manhattan) and a Top-K algorithm to find the nearest vectors.
- **Part 2: K-Means and Approximate Nearest Neighbor (ANN)**: We implemented the K-Means clustering algorithm and used it to build an ANN search algorithm.

The goal of this task was to learn about GPU programming, performance optimization, and profiling. We wrote a report analyzing the implementation, performance, and scalability.

### Task 2: Integration into Distributed System

In this task, we built and optimized a FastAPI-based Retrieval-Augmented Generation (RAG) service. This task was divided into four steps:

- **Step 1: Build a RAG Service**: We created a basic RAG service that combines document retrieval with text generation.
- **Step 2: Performance Testing**: We created a script to test the service with different request rates to measure its performance.
- **Step 3: Optimization**: We implemented a request queue and batch processing mechanism to handle concurrent requests and improve performance.
- **Step 4: Autoscaling and Load Balancing**: We implemented a load balancer and autoscaler to dynamically adjust the number of service replicas based on the request rate.

The goal of this task was to learn about building and deploying end-to-end machine learning systems, measuring system performance, and optimizing for scalability.

### Paper Writing

- We wrote a paper documenting the work on both tasks in the format of a NeurIPS or ICML paper. This can be found: [MLS-report](./MLS-report.pdf).
