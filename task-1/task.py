from tkinter import E
import torch
import cupy as cp
# import triton
import numpy as np
import time
import json
import scipy
from scipy.cluster.vq import vq
from test import testdata_kmeans, testdata_knn, testdata_ann
from time_log_decorater import time_log
from scipy.spatial.distance import cdist
# For Test
from sklearn.cluster import KMeans

# You can create any kernel here
subtract_square = cp.ElementwiseKernel('float32 x, float32 y', 
                                    'float32 z', 
                                    '''
                                    z = (x - y);
                                       z = z * z;                                   
                                    '''
                                    )

subtract_abs = cp.ElementwiseKernel('float32 x, float32 y',
                                    'float32 z',
                                    '''
                                    z = abs(x - y)
                                    ''')

sum_sqrt = cp.ReductionKernel('float32 x', 'float32 y', 'x', 'a + b', 'y = sqrt(a)', '0')

multiply = cp.ElementwiseKernel('float32 x, float32 y', 
                                'float32 z', 
                                '''z = x * y''')

_sum = cp.ReductionKernel('float32 x', 'float32 y', 'x', 'a + b', 'y = a', '0')

square = cp.ElementwiseKernel('float32 x', 'float32 y', '''y = x * x''')

divide = cp.ElementwiseKernel('float32 x, float32 y', 'float32 z', '''z = x / y''')

# def distance_cosine(X, Y, use_kernel=True):
#     if use_kernel:
#         sum_X = sum_sqrt(square(X))
#         sum_Y = sum_sqrt(square(Y))
#         dot = _sum(multiply(X, Y))
#         Z = multiply(sum_X, sum_Y)
#         W = divide(dot, Z)
#         V = 1 - W
#     else:
#         sum_X = cp.linalg.norm(X)
#         sum_Y = cp.linalg.norm(Y)
#         dot = cp.dot(X, Y)
#         W = cp.divide(dot, (sum_X * sum_Y))
#         V = 1 - W
#     return V

def distance_cosine(A, X, use_kernel=True):
    if use_kernel:
        sum_A = sum_sqrt(square(A), axis=1)  # Norm of A (N,)
        sum_X = sum_sqrt(square(X), axis=1)  # Norm of X (1,)
        dot = _sum(multiply(A, X), axis=1)  # Dot product (N,)
        Z = multiply(sum_A, sum_X)  # Multiply norms
        W = divide(dot, Z)  # Normalize dot product
        V = 1 - W  # Cosine distance (N,)
    else:
        sum_A = cp.linalg.norm(A, axis=1)
        sum_X = cp.linalg.norm(X, axis=1)
        dot = cp.sum(A * X, axis=1)
        W = dot / (sum_A * sum_X)
        V = 1 - W
    return V


def distance_cosine_streams(X, Y, use_kernel=True):
    stream1 = cp.cuda.Stream()
    stream2 = cp.cuda.Stream()
    stream3 = cp.cuda.Stream()
    if use_kernel:
        with stream1:
            sum_X = sum_sqrt(square(X))
        with stream2:
            sum_Y = sum_sqrt(square(Y))
        Z = multiply(sum_X, sum_Y)
        with stream3:
            dot = _sum(multiply(X, Y))
        W = divide(dot, Z)
        V = 1 - W
    else:
        with stream1:
            sum_X = cp.linalg.norm(X)
        with stream2:
            sum_Y = cp.linalg.norm(Y)
        with stream3:
            dot = cp.dot(X, Y)
        W = cp.divide(dot, (sum_X * sum_Y))
        V = 1 - W
    return V


# def distance_l2(X, Y, use_kernel=True):
#     if use_kernel:
#         W = subtract_square(X, Y)
#         V = sum_sqrt(W) 
#     else:
#         V = cp.linalg.norm(X - Y) 
#     return V

# def distance_dot(X, Y, use_kernel=True):
#     if use_kernel:
#         Z = multiply(X, Y)
#         W = _sum(Z)
#     else:
#         W = cp.dot(X, Y)
#     return W

# def distance_manhattan(X, Y, use_kernel=True):
#     if use_kernel:
#         Z = subtract_abs(X, Y)
#         U = _sum(Z)
#     else:
#         U = cp.sum(cp.abs(X - Y))
#     return U

def distance_l2(A, X, use_kernel=True):
    if use_kernel:
        W = subtract_square(A, X)  # Element-wise squared difference (N, D)
        V = sum_sqrt(W, axis=1)  # Sum across D and take sqrt (N,)
    else:
        V = cp.linalg.norm(A - X, axis=1)  # GPU-accelerated L2 norm
    return V

def distance_l2_squared(A, X, use_kernel=True):
    if use_kernel:
        W = subtract_square(A, X)  # Element-wise squared difference (N, D)
        U = _sum(W, axis=1)  # Sum across D (N,)
    else:
        U = cp.sum((A - X) ** 2, axis=1)  # Efficiently compute squared L2 distance
    return U

def distance_manhattan(A, X, use_kernel=True):
    if use_kernel:
        Z = subtract_abs(A, X)  # Element-wise absolute difference (N, D)
        U = _sum(Z, axis=1)  # Sum across D (N,)
    else:
        U = cp.sum(cp.abs(A - X), axis=1)  # Use CuPy's optimized sum
    return U

def distance_dot(A, X, use_kernel=True):
    if use_kernel:
        Z = multiply(A, X)  # Element-wise multiplication (N, D)
        W = _sum(Z, axis=1)  # Sum across D (N,)
    else:
        W = cp.sum(A * X, axis=1)  # Efficiently compute dot product
    return W


def distance_cosine_np(X, Y):
    sum_X = np.linalg.norm(X)
    sum_Y = np.linalg.norm(Y)
    dot = np.dot(X, Y)
    W = np.divide(dot, (sum_X * sum_Y))
    V = 1 - W
    return V

def distance_l2_np(X, Y):
    return np.linalg.norm(X - Y, axis=1)

def distance_l2_squared_np(X, Y):
    return np.sum((X - Y) ** 2, axis=1)

def distance_dot_np(X, Y):
    return np.dot(X, Y)

def distance_manhattan_np(X, Y):
    return np.sum(np.abs(X - Y))

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn_cupy_basic(N, D, A, X, K, distance_metric="l2", use_kernel = True):
    """_knn

    Args:
        N (int): Number of vectors
        D (int): Dimension of vectors
        A (list[list[float]]): A collection of vectors(N x D)
        X (list[float]): A specified vector(ie. query vector)
        K (int): topK nearest neighbors to find
        distance_metric (str, optional): _description_. Defaults to "l2".
        use_kernel (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """


    if A.shape != (N, D) or X.shape != (D,):
        raise ValueError("Shape mismatch: A should be (N, D) and X should be (D,)")

    # Compute distances based on chosen metric
        
        # if distance_metric == "cosine":
        #     distances = distance_cosine(A, X, use_kernel)
        # elif distance_metric == "l2":
        #     distances = distance_l2(A, X, use_kernel)
        # elif distance_metric == "dot":
        #     distances = -distance_dot(A, X, use_kernel)
        # elif distance_metric == "manhattan":
        #     distances = distance_manhattan(A, X, use_kernel) 
    if distance_metric == "cosine":
        distances = distance_cosine(A, X[None, :], use_kernel)  # Broadcast X across all rows of A
    elif distance_metric == "l2":
        distances = distance_l2(A, X, use_kernel)  # Apply L2 distance using kernel
    elif distance_metric == "l2_squared":
        distances = distance_l2_squared(A, X, use_kernel)
    elif distance_metric == "dot":
        distances = -distance_dot(A, X[None, :], use_kernel)  # Apply dot product distance
    elif distance_metric == "manhattan":
        distances = distance_manhattan(A, X[None, :], use_kernel)  # Apply Manhattan distance
    else:
        raise ValueError("Unsupported distance metric. Choose from ['l2', 'cosine', 'manhattan', 'dot']")

        # Get the indices of the top K smallest distances
    top_k_indices = cp.argsort(distances)[:K]

    return top_k_indices


def our_knn_np(N, D, A, X, K, distance_metric="l2"):

    if A.shape != (N, D) or X.shape != (D,):
        raise ValueError("Shape mismatch: A should be (N, D) and X should be (D,)")

    # Compute distances based on chosen metric # Comment: Should use matrix multiplication, not multiplying each row iteratively, inefficient process
    if distance_metric == "cosine":
        distances =  np.array(distance_cosine_np(A, X))
    elif distance_metric == "l2":
        distances = np.array(distance_l2_np(A, X))
    elif distance_metric == "l2_squared":
        distances = np.array(distance_l2_squared_np(A, X))
    elif distance_metric == "dot":
        distances = -np.array(distance_dot_np(A, X))
    elif distance_metric == "manhattan":
        distances = np.array(distance_manhattan_np(A, X))
    else:
        raise ValueError("Unsupported distance metric. Choose from ['l2', 'cosine', 'manhattan', 'dot']")

    # Get the indices of the top K smallest distances
    top_k_indices = np.argsort(distances)[:K]

    return top_k_indices




def our_knn_nearest_batch(N, D, A, X, K, batch_size=100000, distance_metric="l2", use_kernel=True):

    if A.shape != (N, D) or X.shape != (D,):
        raise ValueError("Shape mismatch: A should be (N, D) and X should be (D,)")

    top_k_results = []
    top_k_distances = []

    for i in range(0, N, batch_size):
        batch_A = A[i:i+batch_size]  # Extract batch

        if distance_metric == "cosine":
            distances = distance_cosine(batch_A, X[None, :], use_kernel)  # Broadcast X across all rows of A
        elif distance_metric == "l2":
            distances = distance_l2(batch_A, X[None, :], use_kernel)  # Apply L2 distance using kernel
        elif distance_metric == "l2_squared":
            distances = distance_l2_squared(batch_A, X[None, :], use_kernel)
        elif distance_metric == "dot":
            distances = -distance_dot(batch_A, X[None, :], use_kernel)  # Apply dot product distance
        elif distance_metric == "manhattan":
            distances = distance_manhattan(batch_A, X[None, :], use_kernel)  # Apply Manhattan distance
        else:
            raise ValueError("Unsupported distance metric. Choose from ['l2', 'cosine', 'manhattan', 'dot']")

        # Get Top-K from this batch
        batch_top_k = cp.argsort(distances)[:K]
        batch_top_k_distances = distances[batch_top_k]

        # Adjust indices for batch offset
        top_k_results.append(batch_top_k + i)
        top_k_distances.append(batch_top_k_distances)

    # Merge Top-K results from all batches
    top_k_results = cp.concatenate(top_k_results)
    top_k_distances = cp.concatenate(top_k_distances)

    # Get final Top-K across all batches
    final_top_k = cp.argsort(top_k_distances)[:K]
    top_k_indices = top_k_results[final_top_k]
    return top_k_indices 

def our_knn(N, D, A, X, K, gpu= True ,distance_metric="l2", use_kernel = True, batch_size=100000):
    """_knn

    Args:
        N (int): Number of vectors
        D (int): Dimension of vectors
        A (list[list[float]]): A collection of vectors(N x D)
        X (list[float]): A specified vector(ie. query vector)
        K (int): topK nearest neighbors to find
        distance_metric (str, optional): _description_. Defaults to "l2".
        use_kernel (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    if N >= batch_size:
        top_k_indices = our_knn_nearest_batch(N,D,A,X,K,batch_size=batch_size, distance_metric=distance_metric, use_kernel= use_kernel)
    
    else:
        if gpu:
            top_k_indices = our_knn_cupy_basic(N,D,A,X,K, distance_metric = distance_metric, use_kernel = use_kernel)
        else:
            top_k_indices = our_knn_np(N,D,A,X,K, distance_metric=distance_metric)
    
    

    return top_k_indices


# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------
# RAW KERNELS


distance_kernel = cp.RawKernel(r'''
extern "C" __global__
void distance_kernel(const float* A, const float* centroids, 
                                  float* distances, 
                                  const int N, const int K, const int D) {
    int idx = blockIdx.x;  // overall index among N*K distances
    if (idx >= N * K) return;
    
    int n = idx / K;  // index into A
    int k = idx % K;  // index into centroids

    int threadId = threadIdx.x;
    float partial_sum = 0.0f;
    
    int baseA = n * D;
    int baseC = k * D;
    
    for (int d = threadId; d < D; d += blockDim.x) {
        float diff = A[baseA + d] - centroids[baseC + d];
        partial_sum += diff * diff;
    }
    
    extern __shared__ float sdata[];
    sdata[threadId] = partial_sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadId < s) {
            sdata[threadId] += sdata[threadId + s];
        }
        __syncthreads();
    }
    
    if (threadId == 0) {
        distances[idx] = sqrtf(sdata[0]);
    }
}
''', 'distance_kernel')

def vectorized_distance_l2(A, centroids, block_size=128):
    """
    Compute L2 distances between each row in A and each centroid using the fused kernel
    with shared memory.
    
    Parameters:
      A: (N, D) data points.
      centroids: (K, D) centroids.
      block_size: Number of threads per block (defines shared memory size).
      
    Returns:
      distances: (N, K) matrix of Euclidean distances.
    """
    N, D = A.shape
    K = centroids.shape[0]
    total = N * K
    distances = cp.empty(total, dtype=cp.float32)
    
    grid_size = total  
    shared_mem_bytes = block_size * cp.dtype(cp.float32).itemsize
    
    distance_kernel((grid_size,), (block_size,), 
                                 (A, centroids, distances, cp.int32(N), cp.int32(K), cp.int32(D)),
                                 shared_mem=shared_mem_bytes)
    
    return distances.reshape(N, K)

accumulate_centroids_kernel = cp.RawKernel(r'''
extern "C" __global__
void accumulate_centroids_kernel(const float* A, const int* labels, float* new_centroids, 
                                 float* counts, const int N, const int D) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        int label = labels[i];
        for (int d = 0; d < D; d++) {
            atomicAdd(&new_centroids[label * D + d], A[i * D + d]);
        }
        atomicAdd(&counts[label], 1.0f);
    }
}
''', 'accumulate_centroids_kernel')

finalize_centroids_kernel = cp.RawKernel(r'''
extern "C" __global__
void finalize_centroids_kernel(float* new_centroids, const float* old_centroids, 
                               const float* counts, const int K, const int D) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = K * D;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int cluster = i / D;
        if (counts[cluster] > 0.0f) {
            new_centroids[i] = new_centroids[i] / counts[cluster];
        } else {
            new_centroids[i] = old_centroids[i];  // If no points, retain old centroid.
        }
    }
}
''', 'finalize_centroids_kernel')

def update_centroids(A, labels, old_centroids, K):
    """
    Update centroids using the custom accumulation and finalization kernels.
    
    Parameters:
      A: (N, D) data points (CuPy array, float32).
      labels: (N,) cluster labels (CuPy array, int32).
      old_centroids: (K, D) current centroid positions.
      K: Number of clusters.
    
    Returns:
      new_centroids: (K, D) updated centroid positions.
    """
    N, D = A.shape
    new_centroids = cp.zeros((K, D), dtype=cp.float32)
    counts = cp.zeros((K,), dtype=cp.float32)
    
    threads_per_block = 256
    blocks_A = (N + threads_per_block - 1) // threads_per_block
    
    accumulate_centroids_kernel((blocks_A,), (threads_per_block,),
                                (A, labels.astype(cp.int32), new_centroids, counts, cp.int32(N), cp.int32(D)))
    
    total_elements = K * D
    blocks_final = (total_elements + threads_per_block - 1) // threads_per_block
    
    finalize_centroids_kernel((blocks_final,), (threads_per_block,),
                              (new_centroids, old_centroids, counts, cp.int32(K), cp.int32(D)))
    
    return new_centroids

def our_kmeans_raw_kernels(N, D, A, K, block_size=128):
    """
    Perform k-means clustering using:
      - The fused shared-memory distance kernel for computing distances.
      - Custom kernels for updating centroids.
    
    Parameters:
      N (int): Number of data points.
      D (int): Dimension of each data point.
      A (list[list[float]]): Data points.
      K (int): Number of clusters.
      block_size (int): Block size for the distance kernel.
    
    Returns:
      cp.ndarray: Cluster labels for each data point.
    """
    max_iterations = 100
    
    A = cp.asarray(A, dtype=cp.float32)
    indices = cp.random.choice(N, K, replace=False)
    centroids = A[indices, :]
    
    for _ in range(max_iterations):
        distances = vectorized_distance_l2(A, centroids, block_size=block_size)
        labels = cp.argmin(distances, axis=1)
        new_centroids = update_centroids(A, labels, centroids, K)
        
        if cp.allclose(centroids, new_centroids, atol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids
        
    return labels, centroids

# -------------------------------------------------------------------------
# CUPY Basic

def our_kmeans_cupy_basic(N, D, A, K):
    """
    Input:
      N (int): Number of vectors.
      D (int): Dimension of each vector.
      A (list[list[float]]): Collection of vectors (N x D).
      K (int): Number of clusters.
    
    Returns:
      cp.ndarray: Cluster IDs for each vector.
    """
    max_iterations = 100

    A = cp.asarray(A, "float32")
    indices = cp.random.choice(N, K, replace=False)
    centroids = A[indices, :]
    
    for _ in range(max_iterations):
        # Usign Kernel, getting l2 distance
        W = subtract_square(A[:, None, :], centroids[None, :, :])
        distances = sum_sqrt(W, axis=2)
        labels = cp.argmin(distances, axis=1)
        
        new_centroids = cp.zeros_like(centroids)
        cp.add.at(new_centroids, labels, A)
        
        counts = cp.bincount(labels, minlength=K).astype(cp.float32)
        counts = counts.reshape(-1, 1)  # shape (K, 1)
        
        new_centroids = cp.where(counts > 0, new_centroids / counts, centroids)
        
        if cp.allclose(centroids, new_centroids, atol=1e-6):
            centroids = new_centroids
            break
        
        centroids = new_centroids
        
    return labels, centroids

def our_kmeans_cupy_basic_batch(N, D, A, K, batch_size=10000):
    """
    Efficient CuPy K-Means implementation with batched memory-efficient distance computation.
    
    Args:
      N (int): Number of vectors.
      D (int): Dimension of each vector.
      A (cupy.ndarray): Collection of vectors (N x D).
      K (int): Number of clusters.
      batch_size (int): Number of data points to process per batch to reduce memory footprint.
    
    Returns:
      cp.ndarray: Cluster IDs for each vector.
      cp.ndarray: Cluster centers.
    """
    max_iterations = 100

    A = cp.asarray(A, dtype=cp.float32)  # Ensure A is in GPU memory
    indices = cp.random.choice(N, K, replace=False)
    centroids = A[indices, :]

    for _ in range(max_iterations):
        labels = cp.zeros(N, dtype=cp.int32)

        # Process distances in batches to avoid high memory usage
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            A_batch = A[batch_start:batch_end]  # Shape (batch_size, D)

            # Compute squared L2 distance in a memory-efficient way
            dists = (
                cp.sum(A_batch ** 2, axis=1, keepdims=True)
                - 2 * A_batch @ centroids.T
                + cp.sum(centroids ** 2, axis=1)
            )  # Shape (batch_size, K)

            labels[batch_start:batch_end] = cp.argmin(dists, axis=1)

        # Compute new centroids
        new_centroids = cp.zeros_like(centroids)
        counts = cp.zeros(K, dtype=cp.float32)

        # Sum up points in each cluster
        cp.add.at(new_centroids, labels, A)
        cp.add.at(counts, labels, 1)

        # Avoid division by zero
        counts = cp.where(counts > 0, counts, 1)
        new_centroids /= counts[:, None]

        # Convergence check
        if cp.allclose(centroids, new_centroids, atol=1e-6):
            break

        centroids = new_centroids

    return labels, centroids



# -------------------------------------------------------------------------
# NUMPY CPU

def our_kmeans_numpy(N, D, A, K):
    """
    Input:
      N (int): Number of vectors.
      D (int): Dimension of each vector.
      A (list[list[float]]): Collection of vectors (N x D).
      K (int): Number of clusters.
    
    Returns:
      np.ndarray: Cluster IDs for each vector.
    """

    max_iterations = 100

    A = np.asarray(A, dtype=np.float32)
    indices = np.random.choice(N, K, replace=False)
    centroids = A[indices, :]
    
    for _ in range(max_iterations):
        diff = A[:, None, :] - centroids[None, :, :]
        sq_diff = np.sum(diff ** 2, axis=2)
        distances = np.sqrt(sq_diff)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.zeros_like(centroids)
        np.add.at(new_centroids, labels, A)
        
        counts = np.bincount(labels, minlength=K).astype(np.float32)
        counts = counts.reshape(-1, 1)  # shape (K, 1)
        
        new_centroids = np.where(counts > 0, new_centroids / counts, centroids)
        
        if np.allclose(centroids, new_centroids, atol=1e-6):
            centroids = new_centroids
            break
        
        centroids = new_centroids
        
    return labels, centroids

def our_kmeans_numpy_batch(N, D, A, K, batch_size=10000):
    """
    Efficient NumPy K-Means implementation with batched memory-efficient distance computation.

    Args:
      N (int): Number of vectors.
      D (int): Dimension of each vector.
      A (numpy.ndarray): Collection of vectors (N x D).
      K (int): Number of clusters.
      batch_size (int): Number of data points to process per batch to reduce memory footprint.

    Returns:
      np.ndarray: Cluster IDs for each vector.
      np.ndarray: Cluster centers.
    """
    max_iterations = 100

    A = np.asarray(A, dtype=np.float32)  # Ensure A is in CPU memory
    indices = np.random.choice(N, K, replace=False)
    centroids = A[indices, :]

    for _ in range(max_iterations):
        labels = np.zeros(N, dtype=np.int32)

        # Process distances in batches to avoid high memory usage
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            A_batch = A[batch_start:batch_end]  # Shape (batch_size, D)

            # Compute squared L2 distance in a memory-efficient way
            dists = (
                np.sum(A_batch ** 2, axis=1, keepdims=True)
                - 2 * A_batch @ centroids.T
                + np.sum(centroids ** 2, axis=1)
            )  # Shape (batch_size, K)

            labels[batch_start:batch_end] = np.argmin(dists, axis=1)

        # Compute new centroids
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(K, dtype=np.float32)

        # Sum up points in each cluster
        np.add.at(new_centroids, labels, A)
        np.add.at(counts, labels, 1)

        # Avoid division by zero
        counts = np.where(counts > 0, counts, 1)
        new_centroids /= counts[:, None]

        # Convergence check
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break

        centroids = new_centroids

    return labels, centroids

# TODO add return centroids? piazza mentions main function should have specified input outputs.

def dbscan(A, eps, minPts):
    """
    DBSCAN clustering algorithm.
    
    Parameters:
    - A (list[list[float]]): Collection of vectors (N x D).
    - eps: maximum distance between two points to be considered neighbors
    - minPts: minimum number of points to form a dense region (core point)
    
    Returns:
    - centroids: list[list[float]], centroids of each cluster
    - labels: list[int], cluster IDs for each point (-2 for noise, 0+ for clusters)
    """
    labels = np.full(len(A), -1)  # -1 means unvisited/noise
    cluster_id = 0
    
    # Run DBSCAN labeling
    for i in range(len(A)):
        if labels[i] != -1:  # Skip if already processed
            continue
        neighbors = np.where(distance_l2(A, A[i], use_kernel=True) <= eps)[0]
        if len(neighbors) < minPts:  # Not dense enough
            labels[i] = -2  # Noise
        else:
            # Start a new cluster
            labels[i] = cluster_id
            for j in neighbors:
                if labels[j] == -1 or labels[j] == -2:  # Unvisited or noise
                    labels[j] = cluster_id
                    sub_neighbors = np.where(distance_l2(A, A[j], use_kernel=True) <= eps)[0]
                    if len(sub_neighbors) >= minPts:  # Expand if core point
                        neighbors = np.union1d(neighbors, sub_neighbors)
            cluster_id += 1
    
    # Calculate centroids for each cluster
    centroids = []
    for cid in range(cluster_id):  # Iterate over cluster IDs (0, 1, 2, ...)
        cluster_points = A[labels == cid]  # Points in this cluster
        if len(cluster_points) > 0:  # Avoid empty clusters (shouldn't happen, but safety)
            centroid = np.mean(cluster_points, axis=0).tolist()  # Mean along feature axis
            centroids.append(centroid)
    
    # Convert labels to list for return type consistency
    labels = labels.tolist()
    
    return labels, centroids


# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# ------------------------------------------------------------------------------------------------
# IVFPQ Numpy Version
def product_quantization_numpy(D, M, A):
    """
    Performs Product Quantization (PQ) on the dataset.
    """
    assert D % M == 0, "D should be divisible by M"

    sub_dim = D // M
    codebooks = []
    encoded_data = []

    A = np.array(A, dtype=np.float32)
    
    for i in range(M):
        sub_vectors = A[:, i * sub_dim : (i + 1) * sub_dim]
        labels, centroids = our_kmeans_numpy_batch(sub_vectors.shape[0], sub_dim, sub_vectors, 256)
        codebooks.append(centroids)
        encoded_data.append(labels)

    return codebooks, np.array(encoded_data, dtype=np.int32).T

def ivfpq_index_numpy(N, D, A, num_clusters=100, M=8):
    """
    Indexes dataset A using IVF and PQ.
    """
    A = np.array(A, dtype=np.float32)
    cluster_ids, cluster_centers = our_kmeans_numpy_batch(N, D, A, num_clusters)
    
    ivf_lists = {i: [] for i in range(num_clusters)}
    for i, cid in enumerate(cluster_ids):
        ivf_lists[int(cid)].append(i)
    
    codebooks, encoded_data = product_quantization_numpy(D, M, A)
    
    return ivf_lists, cluster_centers, codebooks, encoded_data
@time_log
def search_ivfpq_numpy(X, ivf_lists, cluster_centers, codebooks, encoded_data, A, K=50, num_probe=10, M=8, D=64, candidate_factor=2):
    """
    Performs an IVFPQ ANN search with batched PQ distance computation + final re-ranking.
    """
    start_time = time.perf_counter()

    # 1) Find the closest clusters using IVF
    X = np.array(X, dtype=np.float32)
    
    cluster_distances = distance_l2_np(cluster_centers, X)
    closest_clusters = np.argsort(cluster_distances)[:num_probe]
    
    # gather candidates
    candidates_list = []
    for cid in closest_clusters:
        candidates_list.extend(ivf_lists[int(cid)])
    
    # convert candidates to numpy array
    candidates = np.array(candidates_list, dtype=np.int32)
    if candidates.size == 0:
        # no candidates found
        return []
    
    sub_dim = D // M

    # 2) Batching PQ distance computation
    big_reconstructed = np.zeros((candidates.shape[0], M, sub_dim), dtype=np.float32)
    
    for i in range(M):
        code_ids = encoded_data[candidates, i]
        big_reconstructed[:, i, :] = codebooks[i][code_ids]
    
    # build query sub-vectors
    X_subvectors = np.zeros((M, sub_dim), dtype=np.float32)
    for i in range(M):
        X_subvectors[i] = X[i * sub_dim : (i + 1) * sub_dim]
    
    diff = big_reconstructed - X_subvectors[None, :, :]
    dist_vector = np.sqrt(np.sum(diff**2, axis=(1, 2)))
    
    # sort by approximate PQ distance
    sorted_indices = np.argsort(dist_vector)
    distances_all = [(int(candidates[i]), float(dist_vector[i])) for i in range(len(candidates))]
    distances_all.sort(key=lambda x: x[1])
    
    # 3) Final Re-Ranking (Exact Distance) on top subset
    refine_size = candidate_factor * K
    approx_top = distances_all[:refine_size]
    
    exact_reranked = []
    query_full_unnorm = X * (np.linalg.norm(X) + 1e-6)
    for (idx, _) in approx_top:
        data_unnorm = A[idx] * (np.linalg.norm(A[idx]) + 1e-6)
        diff_full = data_unnorm - query_full_unnorm
        dist = np.linalg.norm(diff_full)
        exact_reranked.append((idx, dist))
    
    exact_reranked.sort(key=lambda x: x[1])
    
    end_time = time.perf_counter()
    print(f"Query Time (Batched + Re-Rank): {(end_time - start_time) * 1000:.3f} ms")
    
    return [idx for idx, _ in exact_reranked[:K]]


def our_ann_numpy(N, D, A, X, K):
    """
    IVFPQ ANN algorithm wrapper.
    """
    num_clusters = 200
    M = 8  # Number of PQ subspaces
    
    ivf_lists, cluster_centers, codebooks, encoded_data = ivfpq_index_numpy(N, D, A, num_clusters, M)
    top_k_indices = search_ivfpq_numpy(
        X, ivf_lists, cluster_centers, codebooks, encoded_data, A, 
        K=K, num_probe=40, M=M, D=D, candidate_factor=2
    )
    
    return top_k_indices

# ------------------------------------------------------------------------------------------------
# IVFPQ CuPy Version
def product_quantization_cupy(D, M, A):
    """
    Performs Product Quantization (PQ) on the dataset.
    """
    assert D % M == 0, "D should be divisible by M"
    
    sub_dim = D // M
    codebooks = []
    encoded_data = []
    
    A = cp.array(A, dtype=cp.float32)
    
    for i in range(M):
        sub_vectors = A[:, i * sub_dim : (i + 1) * sub_dim]
        labels, centroids = our_kmeans_cupy_basic_batch(sub_vectors.shape[0], sub_dim, sub_vectors, 256)
        codebooks.append(centroids)
        encoded_data.append(labels)
    
    return codebooks, cp.array(encoded_data, dtype=cp.int32).T

def ivfpq_index_cupy(N, D, A, num_clusters=100, M=8):
    """
    Indexes dataset A using IVF and PQ.
    """
    A = cp.array(A, dtype=cp.float32)
    cluster_ids, cluster_centers = our_kmeans_cupy_basic_batch(N, D, A, num_clusters)
    cluster_ids = cluster_ids.get()  # Convert to NumPy to use as dictionary keys
    
    ivf_lists = {i: [] for i in range(num_clusters)}
    for i, cid in enumerate(cluster_ids):
        ivf_lists[cid].append(i)
    
    codebooks, encoded_data = product_quantization_cupy(D, M, A)
    
    return ivf_lists, cluster_centers, codebooks, encoded_data

@time_log
def search_ivfpq_cupy(X, ivf_lists, cluster_centers, codebooks, encoded_data, A, K=50, num_probe=10, M=8, D=64, candidate_factor=2):
    """
    Performs an IVFPQ ANN search.
    """

    X = cp.array(X, dtype=cp.float32)

    cluster_distances = distance_l2(cluster_centers, X, use_kernel=False)
    closest_clusters = cp.argsort(cluster_distances)[:num_probe]
    
    # gather candidates
    candidates_list = []
    for cid in cp.asnumpy(closest_clusters):
        candidates_list.extend(ivf_lists[int(cid)])

    # convert candidates to cupy array
    candidates = cp.array(candidates_list, dtype=cp.int32)
    if candidates.size == 0:
        # no candidates found
        return []
    
    sub_dim = D // M
    # 2) Batching PQ distance computation
    # build big_reconstructed shape: (len(candidates), M, sub_dim)
    big_reconstructed = cp.zeros((candidates.shape[0], M, sub_dim), dtype=cp.float32)

    for i in range(M):
        code_ids = encoded_data[candidates, i]  # shape (num_candidates,)
        big_reconstructed[:, i, :] = codebooks[i][code_ids]

    # build query sub-vectors shape (M, sub_dim)
    X_subvectors = cp.zeros((M, sub_dim), dtype=cp.float32)
    for i in range(M):
        X_subvectors[i] = X[i * sub_dim : (i + 1) * sub_dim]

    # diff shape: (num_candidates, M, sub_dim)
    diff = big_reconstructed - X_subvectors[None, :, :]
    dist_vector = cp.sqrt(cp.sum(diff**2, axis=(1, 2)))  # shape (num_candidates,)

    # sort by approximate PQ distance
    dist_vector_cpu = cp.asnumpy(dist_vector)
    candidates_cpu = cp.asnumpy(candidates)

    distances_all = [(int(candidates_cpu[i]), float(dist_vector_cpu[i])) for i in range(len(candidates_cpu))]
    # we only actually need to sort them all if we plan final re-ranking on top subset
    distances_all.sort(key=lambda x: x[1])
    
    # ---------------------
    # Final Re-Ranking (Exact Distance)
    refine_size = candidate_factor * K  # e.g., 2*K or 4*K
    approx_top = distances_all[:refine_size]
    
    # Convert A to Cupy if it's not already
    if not isinstance(A, cp.ndarray):
        A = cp.array(A, dtype=cp.float32)
    
    # We'll compute exact L2 distances in the original D=64 space
    exact_reranked = []
    query_full = cp.array(X, copy=True)  # shape (D,)
    query_full_unnorm = query_full * (cp.linalg.norm(X) + 1e-6)  
    # If your data A was normalized, you'd keep it the same.
    
    for (idx, _) in approx_top:
        data_unnorm = A[idx] * cp.linalg.norm(A[idx])
        diff = data_unnorm - query_full_unnorm
        dist = cp.linalg.norm(diff)
        exact_reranked.append((idx, dist))
    
    exact_reranked.sort(key=lambda x: x[1])
    

    return [idx for idx, _ in exact_reranked[:K]]

def our_ann_cupy_basic(N, D, A, X, K):
    """
    IVFPQ ANN algorithm wrapper.
    """
    num_clusters = 200
    M = 8 # Number of PQ subspaces
    
    ivf_lists, cluster_centers, codebooks, encoded_data = ivfpq_index_cupy(N, D, A, num_clusters, M)
    top_k_indices = search_ivfpq_cupy(X, ivf_lists, cluster_centers, codebooks, encoded_data, A, K, num_probe=40, M=M, D=D)
    
    return top_k_indices

# ------------------------------------------------------------------------------------------------
# IVFPQ Raw Kernerls Version
def product_quantization_raw_kernerls(D, M, A):
    """
    Performs Product Quantization (PQ) on the dataset.
    """
    assert D % M == 0, "D should be divisible by M"
    
    sub_dim = D // M
    codebooks = []
    encoded_data = []
    
    A = cp.array(A, dtype=cp.float32)
    
    for i in range(M):
        sub_vectors = A[:, i * sub_dim : (i + 1) * sub_dim]
        labels, centroids = our_kmeans_cupy_basic_batch(sub_vectors.shape[0], sub_dim, sub_vectors, 256)
        # labels, centroids = our_kmeans_raw_kernels(sub_vectors.shape[0], sub_dim, sub_vectors, 256)
        codebooks.append(centroids)
        encoded_data.append(labels)
    
    return codebooks, cp.array(encoded_data, dtype=cp.int32).T

def ivfpq_index_raw_kernerls(N, D, A, num_clusters=100, M=8):
    """
    Indexes dataset A using IVF and PQ.
    """
    A = cp.array(A, dtype=cp.float32)
    cluster_ids, cluster_centers = our_kmeans_cupy_basic_batch(N, D, A, num_clusters)
    # cluster_ids, cluster_centers = our_kmeans_raw_kernels(N, D, A, num_clusters)
    cluster_ids = cluster_ids.get()  # Convert to NumPy to use as dictionary keys
    
    ivf_lists = {i: [] for i in range(num_clusters)}
    for i, cid in enumerate(cluster_ids):
        ivf_lists[cid].append(i)
    
    codebooks, encoded_data = product_quantization_raw_kernerls(D, M, A)
    
    return ivf_lists, cluster_centers, codebooks, encoded_data

@time_log
def search_ivfpq_raw_kernels(X, ivf_lists, cluster_centers, codebooks, encoded_data, A, K=50, num_probe=10, M=8, D=64, candidate_factor=2):
    """
    Performs an IVFPQ ANN search.
    """

    X = cp.array(X, dtype=cp.float32)

    cluster_distances = distance_l2(cluster_centers, X, use_kernel=True)
    closest_clusters = cp.argsort(cluster_distances)[:num_probe]
    
    # gather candidates
    candidates_list = []
    for cid in cp.asnumpy(closest_clusters):
        candidates_list.extend(ivf_lists[int(cid)])

    # convert candidates to cupy array
    candidates = cp.array(candidates_list, dtype=cp.int32)
    if candidates.size == 0:
        # no candidates found
        return []
    
    sub_dim = D // M
    # 2) Batching PQ distance computation
    # build big_reconstructed shape: (len(candidates), M, sub_dim)
    big_reconstructed = cp.zeros((candidates.shape[0], M, sub_dim), dtype=cp.float32)

    for i in range(M):
        code_ids = encoded_data[candidates, i]  # shape (num_candidates,)
        big_reconstructed[:, i, :] = codebooks[i][code_ids]

    # build query sub-vectors shape (M, sub_dim)
    X_subvectors = cp.zeros((M, sub_dim), dtype=cp.float32)
    for i in range(M):
        X_subvectors[i] = X[i * sub_dim : (i + 1) * sub_dim]

    # diff shape: (num_candidates, M, sub_dim)
    diff = big_reconstructed - X_subvectors[None, :, :]
    dist_vector = cp.sqrt(cp.sum(diff**2, axis=(1, 2)))  # shape (num_candidates,)

    # sort by approximate PQ distance
    dist_vector_cpu = cp.asnumpy(dist_vector)
    candidates_cpu = cp.asnumpy(candidates)

    distances_all = [(int(candidates_cpu[i]), float(dist_vector_cpu[i])) for i in range(len(candidates_cpu))]
    # we only actually need to sort them all if we plan final re-ranking on top subset
    distances_all.sort(key=lambda x: x[1])
    
    # ---------------------
    # Final Re-Ranking (Exact Distance)
    refine_size = candidate_factor * K  # e.g., 2*K or 4*K
    approx_top = distances_all[:refine_size]
    
    # Convert A to Cupy if it's not already
    if not isinstance(A, cp.ndarray):
        A = cp.array(A, dtype=cp.float32)
    
    # We'll compute exact L2 distances in the original D=64 space
    exact_reranked = []
    query_full = cp.array(X, copy=True)  # shape (D,)
    query_full_unnorm = query_full * (cp.linalg.norm(X) + 1e-6)  
    # If your data A was normalized, you'd keep it the same.
    
    for (idx, _) in approx_top:
        data_unnorm = A[idx] * cp.linalg.norm(A[idx])
        diff = data_unnorm - query_full_unnorm
        dist = cp.linalg.norm(diff)
        exact_reranked.append((idx, dist))
    
    exact_reranked.sort(key=lambda x: x[1])
    

    return [idx for idx, _ in exact_reranked[:K]]

def our_ann_raw_kernerls(N, D, A, X, K):
    """
    IVFPQ ANN algorithm wrapper.
    """
    num_clusters = 200
    M = 8 # Number of PQ subspaces
    
    ivf_lists, cluster_centers, codebooks, encoded_data = ivfpq_index_raw_kernerls(N, D, A, num_clusters, M)
    top_k_indices = search_ivfpq_raw_kernels(X, ivf_lists, cluster_centers, codebooks, encoded_data, A, K, num_probe=40, M=M, D=D)
    
    return top_k_indices

# ------------------------------------------------------------------------------------------------
# Wrapping for test
@time_log
def our_knn(N, D, A, X, K):
    global KNN_IMPLEMENTATIONS
    
    if KNN_IMPLEMENTATIONS == "raw_kernerls":
        # top_k_indices = our_knn_nearest_batch(N, D, A, X, K, batch_size=100000, distance_metric="l2", use_kernel=True)
        top_k_indices = our_knn_cupy_basic(N, D, A, X, K, distance_metric="l2", use_kernel = True)
    elif KNN_IMPLEMENTATIONS == "cupy_basic":    
        top_k_indices = our_knn_cupy_basic(N, D, A, X, K, distance_metric="l2", use_kernel = False)
    elif KNN_IMPLEMENTATIONS == "numpy":
        top_k_indices = our_knn_np(N, D, A, X, K, distance_metric="l2")

    return top_k_indices

@time_log
def our_kmeans(N, D, A, K):
    global KMEANS_IMPLEMENTATIONS
    
    if KMEANS_IMPLEMENTATIONS == "raw_kernerls":
        labels, centroids = our_kmeans_raw_kernels(N, D, A, K, block_size=128)
    elif KMEANS_IMPLEMENTATIONS == "cupy_basic":    
        labels, centroids = our_kmeans_cupy_basic(N, D, A, K)
    elif KMEANS_IMPLEMENTATIONS == "numpy":
        labels, centroids = our_kmeans_numpy(N, D, A, K)
    elif KMEANS_IMPLEMENTATIONS == "dbscan":
        labels, centorids = dbscan(A, 0.5, 5)
    return labels


@time_log
def our_ann(N, D, A, X, K):
    global ANN_IMPLEMENTATIONS

    if ANN_IMPLEMENTATIONS == "numpy":
        top_k_indices = our_ann_numpy(N, D, A, X, K)
    elif ANN_IMPLEMENTATIONS == "cupy_basic":
        top_k_indices = our_ann_cupy_basic(N, D, A, X, K)
    elif ANN_IMPLEMENTATIONS == "raw_kernerls":
        top_k_indices = our_ann_raw_kernerls(N, D, A, X, K)
    
    return top_k_indices


# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

def test_cosine(D=2, use_kernel=True):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time.time()
    ours = distance_cosine(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = 1 - torch.cosine_similarity(torch.tensor(X, dtype=float), torch.tensor(Y, dtype=float), dim=0).item()
    assert cp.isclose([ours], [gold], rtol=1e-06, atol=1e-6)
    # print("Execution Time: {}".format(end - start))
    return end-start

def test_cosine_streams(D=2, use_kernel=True):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time.time()
    ours = distance_cosine_streams(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = 1 - torch.cosine_similarity(torch.tensor(X, dtype=float), torch.tensor(Y, dtype=float), dim=0).item()
    assert cp.isclose([ours], [gold], rtol=1e-06, atol=1e-6)
    # print("Execution Time: {}".format(end - start))
    return end-start

def test_l2(D=2, use_kernel=True):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time.time()
    ours = distance_l2(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = cp.linalg.norm(X - Y)
    assert cp.isclose([ours], [gold])
    # print("Execution Time: {}".format(end - start))
    return end-start

def test_dot(D=2, use_kernel=True):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time.time()
    ours = distance_dot(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = cp.dot(X, Y)
    assert cp.isclose([ours], [gold])
    # print("Execution Time: {}".format(end - start))
    return end-start

def test_manhattan(D=2, use_kernel=True):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start = time.time()
    ours = distance_manhattan(X, Y, use_kernel=use_kernel)
    end = time.time()
    gold = scipy.spatial.distance.cityblock(X.get(), Y.get())
    assert cp.isclose([ours], [gold])
    # print("Execution Time: {}".format(end - start))
    return end-start

def test_cosine_gpu_vs_cpu(D=2, use_kernel=False):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start_gpu = time.time()
    _ = distance_cosine(X, Y, use_kernel=use_kernel)
    end_gpu = time.time()
    X, Y = np.array(X.get(), dtype=np.float32), np.array(Y.get(), dtype=np.float32)
    start_cpu = time.time()
    _ = distance_cosine_np(X, Y)
    end_cpu = time.time()
    # print(f"Cosine (GPU): {end_gpu-start_gpu}")
    # print(f"Cosine (CPU): {end_cpu-start_cpu}")
    # print(f"Cosine (CPU - GPU): {(end_cpu-start_cpu)  - (end_gpu-start_gpu)}")
    return end_gpu-start_gpu, end_cpu-start_cpu

def test_l2_gpu_vs_cpu(D=2, use_kernel=False):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start_gpu = time.time()
    _ = distance_l2(X, Y, use_kernel=use_kernel)
    end_gpu = time.time()
    X, Y = np.array(X.get(), dtype=np.float32), np.array(Y.get(), dtype=np.float32)
    start_cpu = time.time()
    _ = distance_l2_np(X, Y)
    end_cpu = time.time()
    # print(f"L2 (GPU): {end_gpu-start_gpu}")
    # print(f"L2 (CPU): {end_cpu-start_cpu}")
    # print(f"L2 (CPU - GPU): {(end_cpu-start_cpu)  - (end_gpu-start_gpu)}")
    return end_gpu-start_gpu, end_cpu-start_cpu

def test_dot_gpu_vs_cpu(D=2, use_kernel=False):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start_gpu = time.time()
    _ = distance_dot(X, Y, use_kernel=use_kernel)
    end_gpu = time.time()
    X, Y = np.array(X.get(), dtype=np.float32), np.array(Y.get(), dtype=np.float32)
    start_cpu = time.time()
    _ = distance_dot_np(X, Y)
    end_cpu = time.time()
    # print(f"Dot (GPU): {end_gpu-start_gpu}")
    # print(f"Dot (CPU): {end_cpu-start_cpu}")
    # print(f"Dot (CPU - GPU): {(end_cpu-start_cpu)  - (end_gpu-start_gpu)}")
    return end_gpu-start_gpu, end_cpu-start_cpu

def test_manhattan_gpu_vs_cpu(D=2, use_kernel=False):
    X, Y = cp.random.randn(D, dtype=cp.float32), cp.random.randn(D, dtype=cp.float32)
    start_gpu = time.time()
    _ = distance_manhattan(X, Y, use_kernel=use_kernel)
    end_gpu = time.time()
    X, Y = np.array(X.get(), dtype=np.float32), np.array(Y.get(), dtype=np.float32)
    start_cpu = time.time()
    _ = distance_manhattan_np(X, Y)
    end_cpu = time.time()
    # print(f"Manhattan (GPU): {end_gpu-start_gpu}")
    # print(f"Manhattan (CPU): {end_cpu-start_cpu}")
    # print(f"Manhattan (CPU - GPU): {(end_cpu-start_cpu)  - (end_gpu-start_gpu)}")
    return end_gpu-start_gpu, end_cpu-start_cpu

# Example
def test_kmeans():
    N, D, A, K = testdata_kmeans("")
    start = time.time()
    kmeans_result = our_kmeans_numpy(N, D, A, K)
    end = time.time()
    print("numpy:", end - start)
    # print(kmeans_result)
    
    start = time.time()
    kmeans_result = our_kmeans_cupy_basic(N, D, A, K)
    end = time.time()
    print("cupy basic:", end - start)
    # print(kmeans_result)
    
    start = time.time()
    kmeans_result = our_kmeans_raw_kernels(N, D, A, K)
    end = time.time()
    print("cupy raw kernels:", end - start)
    # print(kmeans_result)
    

def test_knn_numpy():
    N, D, A, X, K = testdata_knn("")

    A = np.asarray(A)
    X = np.asarray(X)
    
    # Run the our_ann function
    top_k_indices_np = our_knn_np(N, D, A, X, K)
    
    top_k_indices_np = cp.asnumpy(top_k_indices_np)
    
    # Check the length of the result
    assert len(top_k_indices_np) == K, f"Expected {K} indices, but got {len(top_k_indices_np)}"
    
    # Check if the indices are within the valid range
    assert np.all(top_k_indices_np < N), "Some indices are out of range"
    
    print("top_k_indices_np.shape:", top_k_indices_np.shape)
    print("top_k_indices_np:", top_k_indices_np)
    
    print("Test passed!")

    return top_k_indices_np
    

    
def test_our_ann():
    # Generate test data
    N, D, A, X, K = testdata_ann("")
    
    # Convert data to CuPy arrays
    A_cp = np.asarray(A)
    X_cp = np.asarray(X)
    
    # Run the our_ann function
    top_k_indices_np = our_ann(N, D, A_cp, X_cp, K)
    
    # Convert the result back to NumPy for assertion
    top_k_indices_np = cp.asnumpy(top_k_indices_np)
    
    # Check the length of the result
    assert len(top_k_indices_np) == K, f"Expected {K} indices, but got {len(top_k_indices_np)}"
    
    # Check if the indices are within the valid range
    assert np.all(top_k_indices_np < N), "Some indices are out of range"
    
    print("top_k_indices_np.shape:", top_k_indices_np.shape)
    print("top_k_indices_np:", top_k_indices_np)
    
    print("Test passed!")
    


def test_our_ann_IVFPQ_numpy():
    # Generate test data
    N, D, A, X, K, = testdata_ann("")
    
    A = np.asarray(A)
    X = np.asarray(X)
    
    
    # Run the our_ann_IVFPQ function
    top_k_indices_np = our_ann(N, D, A, X, K)

    top_k_indices_np = cp.asnumpy(top_k_indices_np)
    
    # Check the length of the result
    assert len(top_k_indices_np) == K, f"Expected {K} indices, but got {len(top_k_indices_np)}"
    
    # Check if the indices are within the valid range
    assert np.all(top_k_indices_np < N), "Some indices are out of range"
    
    print("top_k_indices_np.shape:", top_k_indices_np.shape)
    print("top_k_indices_np:", top_k_indices_np)
    
    print("Test passed!")
    return top_k_indices_np
    
def test_our_ann_IVFPQ_cupy_basic():
    # Generate test data
    N, D, A, X, K, M, n_probe = testdata_ann("")
    
    # Convert data to CuPy arrays
    A_cp = cp.asarray(A)
    X_cp = cp.asarray(X)
    
    # Run the our_ann_IVFPQ function
    top_k_indices = our_ann(N, D, A_cp, X_cp, K)
    
    # Convert the result back to NumPy for assertion
    top_k_indices_np = cp.asnumpy(top_k_indices)
    print("top_k_indices_np:", top_k_indices_np)
    # Check the length of the result
    assert len(top_k_indices_np) == K, f"Expected {K} indices, but got {len(top_k_indices_np)}"
    
    # Check if the indices are within the valid range
    assert np.all(top_k_indices_np < N), "Some indices are out of range"
    
    print("top_k_indices_np.shape:", top_k_indices_np.shape)
    print("top_k_indices_np:", top_k_indices_np)
    
    print("Test passed!")
    return top_k_indices_np

def test_recall_rate_numpy():
    # Generate test data
    N, D, A, X, K = testdata_ann("")
    
    A_cp = np.asarray(A)
    X_cp = np.asarray(X)
    
    # Ensure inputs are normalized
    A = A / np.linalg.norm(A, axis=1, keepdims=True)  # Normalize database
    X = X / np.linalg.norm(X)  # Normalize query
    
    # Run the KNN algorithm
    knn_result_np = our_knn(N, D, A_cp, X_cp, K)
    
    
    # Run the ANN algorithm
    ann_result_np = our_ann(N, D, A_cp, X_cp, K)


    
    # Calculate recall rate
    recall = recall_rate(knn_result_np, ann_result_np, K)

    
    print(f"Recall rate: {recall:.2f}")
    
def test_recall_rate_cupy_basic():
    # Generate test data
    N, D, A, X, K = testdata_ann("")
    
    # Convert data to CuPy arrays
    A = cp.asarray(A, dtype=cp.float32)
    X = cp.asarray(X, dtype=cp.float32)
    
    # Ensure inputs are normalized
    A = A / cp.linalg.norm(A, axis=1, keepdims=True)  # Normalize database
    X = X / cp.linalg.norm(X)  # Normalize query
    
    
    # Run the KNN algorithm
    knn_result = our_knn(N, D, A, X, K)
    knn_result_np = knn_result.get()
    

    # Run the ANN algorithm
    ann_result = our_ann(N, D, A, X, K)
    ann_result_np = ann_result
    
    # Calculate recall rate
    recall = recall_rate(knn_result_np, ann_result_np, K)

    
    print(f"Recall rate: {recall:.2f}")
    
def test_recall_rate_raw_kernerls():
    # Generate test data
    N, D, A, X, K = testdata_ann("")
    
    # Convert data to CuPy arrays
    A = cp.asarray(A, dtype=cp.float32)
    X = cp.asarray(X, dtype=cp.float32)
    
    # Ensure inputs are normalized
    A = A / cp.linalg.norm(A, axis=1, keepdims=True)  # Normalize database
    X = X / cp.linalg.norm(X)  # Normalize query
    
    # Run the KNN algorithm
    knn_result = our_knn(N, D, A, X, K)
    knn_result_np = knn_result.get()
    

    # Run the ANN algorithm
    ann_result = our_ann(N, D, A, X, K)
    ann_result_np = ann_result
    


    # Calculate recall rate
    recall = recall_rate(knn_result_np, ann_result_np, K)

    
    print(f"Recall rate: {recall:.2f}")
    
    
def recall_rate(knn_result, ann_result, K):
    """
    Calculate the recall rate of two lists
    knn_result[K]: The top K nearest vectors ID from KNN
    ann_result[K]: The top K nearest vectors ID from ANN
    
    Returns:
    float: Recall rate
    """
    return len(set(knn_result) & set(ann_result)) / K


if __name__ == "__main__":
    ### Test Ann
    # KMEANS_IMPLEMENTATION = "numpy" # or "raw_kernels", "cupy_basic", "numpy", or, "dbscan"
    
    
    # for _ in range(10):
    #     ANN_IMPLEMENTATIONS = "numpy"  
    #     KNN_IMPLEMENTATIONS = "numpy" 
    #     test_recall_rate_numpy()
    # for _ in range(10):
    #     ANN_IMPLEMENTATIONS = "cupy_basic"  
    #     KNN_IMPLEMENTATIONS = "cupy_basic" 
    #     test_recall_rate_cupy_basic()
    for _ in range(10):
        KNN_IMPLEMENTATIONS = "raw_kernerls"
        ANN_IMPLEMENTATIONS = "raw_kernerls"
        test_recall_rate_raw_kernerls()
        
    # test_recall_rate_sim_cupy()
    # test_knn_numpy()
    
    
    
    # ### Test Distance Functions
    # dimensions = [2, 2**15]
    # N = 30
    # print(f"N={N}")
    # print("Testing Distance Functions")
    # for D in dimensions:
    #     print(f"Dimension: {D}")
    #     cosine_kernel_list, cosine_api_list = cp.empty(N), cp.empty(N)
    #     cosine_kernel_stream_list, cosine_api_stream_list = cp.empty(N), cp.empty(N)
    #     l2_kernel_list, l2_api_list = cp.empty(N), cp.empty(N)
    #     dot_kernel_list, dot_api_list = cp.empty(N), cp.empty(N)
    #     manhattan_kernel_list, manhattan_api_list = cp.empty(N), cp.empty(N)
    #     for i in range(N):
    #         cosine_kernel = test_cosine(D)
    #         cosine_kernel_list[i] = cosine_kernel
    #         cosine_api = test_cosine(D, use_kernel=False)
    #         cosine_api_list[i] = cosine_api
    #         cosine_kernel_stream = test_cosine_streams(D, use_kernel=True)
    #         cosine_kernel_stream_list[i] = cosine_kernel_stream
    #         cosine_api_stream = test_cosine_streams(D, use_kernel=False)
    #         cosine_api_stream_list[i] = cosine_api_stream
    #         l2_kernel = test_l2(D)
    #         l2_kernel_list[i] = l2_kernel
    #         l2_api = test_l2(D, use_kernel=False)
    #         l2_api_list[i] = l2_api
    #         dot_kernel = test_dot(D)
    #         dot_kernel_list[i] = dot_kernel
    #         dot_api = test_dot(D, use_kernel=False)
    #         dot_api_list[i] = dot_api
    #         manhattan_kernel = test_manhattan(D)
    #         manhattan_kernel_list[i] = manhattan_kernel
    #         manhattan_api = test_manhattan(D, use_kernel=False)
    #         manhattan_api_list[i] = manhattan_api
    #     print("----------------------------------------")
    #     print("Absolute Runtime Values (API)")
    #     print(f"Cosine (Stream): {cp.median(cosine_api_stream_list)}")
    #     print(f"Cosine (Without Stream): {cp.median(cosine_api_list)}")
    #     print(f"L2: {cp.median(l2_api_list)}")
    #     print(f"Dot: {cp.median(dot_api_list)}")
    #     print(f"Manhattan: {cp.median(manhattan_api_list)}")
    #     print("----------------------------------------")
    #     print("Absolute Runtime Values (Kernel)")
    #     print(f"Cosine (Stream): {cp.median(cosine_kernel_stream_list)}")
    #     print(f"Cosine (Without Stream): {cp.median(cosine_kernel_list)}")
    #     print(f"L2: {cp.median(l2_kernel_list)}")
    #     print(f"Dot: {cp.median(dot_kernel_list)}")
    #     print(f"Manhattan: {cp.median(manhattan_kernel_list)}")
    #     print("----------------------------------------")
    #     print("Differences in Speed (Positive means API is faster than Kernel)")
    #     print(f"Cosine Difference: {cp.median(cosine_kernel_list) - cp.median(cosine_api_list)}")
    #     print(f"Cosine Difference (Streams): {cp.median(cosine_kernel_stream_list) - cp.median(cosine_api_stream_list)}")
    #     print(f"L2 Difference: {cp.median(l2_kernel_list) - cp.median(l2_api_list)}")
    #     print(f"Dot Difference: {cp.median(dot_kernel_list) - cp.median(dot_api_list)}")
    #     print(f"Manhattan Difference: {cp.median(manhattan_kernel_list) - cp.median(manhattan_api_list)}")
    #     print("----------------------------------------")
    # print(f"Testing Differences Between CPU and GPU")
    # for D in dimensions:
    #     print(f"Dimension: {D}")
    #     cosine_gpu, cosine_cpu = cp.empty(N), cp.empty(N)
    #     l2_gpu, l2_cpu = cp.empty(N), cp.empty(N)
    #     dot_gpu, dot_cpu = cp.empty(N), cp.empty(N)
    #     manhattan_gpu, manhattan_cpu = cp.empty(N), cp.empty(N)
    #     for i in range(N):
    #         gpu, cpu = test_cosine_gpu_vs_cpu(D) # diff = cpu - gpu
    #         cosine_gpu[i] = gpu
    #         cosine_cpu[i] = cpu
    #         gpu, cpu = test_l2_gpu_vs_cpu(D)
    #         l2_gpu[i] = gpu
    #         l2_cpu[i] = cpu
    #         gpu, cpu = test_dot_gpu_vs_cpu(D)
    #         dot_gpu[i] = gpu
    #         dot_cpu[i] = cpu
    #         gpu, cpu = test_manhattan_gpu_vs_cpu(D)
    #         manhattan_gpu[i] = gpu
    #         manhattan_cpu[i] = cpu
    #     print(f"Cosine CPU: {cp.median(cosine_cpu)}")
    #     print(f"Cosine GPU: {cp.median(cosine_gpu)}")
    #     print(f"Cosine CPU - GPU: {(cp.median(cosine_cpu) - cp.median(cosine_gpu)).item()}")
    #     print(f"L2 CPU: {cp.median(l2_cpu)}")
    #     print(f"L2 GPU: {cp.median(l2_gpu)}")
    #     print(f"L2 CPU - GPU: {(cp.median(l2_cpu) - cp.median(l2_gpu)).item()}")
    #     print(f"Dot CPU: {cp.median(dot_cpu)}")
    #     print(f"Dot GPU: {cp.median(dot_gpu)}")
    #     print(f"Dot CPU - GPU: {(cp.median(dot_cpu) - cp.median(dot_gpu)).item()}")
    #     print(f"Manhattan CPU: {cp.median(cosine_cpu)}")
    #     print(f"Manhattan GPU: {cp.median(manhattan_gpu)}")
    #     print(f"Manhattan CPU - GPU: {(cp.median(manhattan_cpu) - cp.median(manhattan_gpu)).item()}")
    #     print("----------------------------------------")
    
    ### Test KNN

    # # Set parameters
    # N, D, K = 4000000, 2, 10

    # # Generate random dataset
    # A_gpu = cp.random.randn(N, D).astype(cp.float32)  # GPU array
    # X_gpu = cp.random.randn(D).astype(cp.float32)

    # # Test with 4M vectors
    # for metric in ["l2", "cosine", "dot", "manhattan"]:
    #     print("running")
    #     top_k_indices = our_knn_nearest_batch(N, D, A_gpu, X_gpu, K, distance_metric=metric, use_kernel=True)
    #     print(f"Top {K} nearest neighbors using {metric} distance:", cp.asnumpy(top_k_indices))

    # # Convert to NumPy for CPU testing
    # A_cpu = cp.asnumpy(A_gpu)
    # X_cpu = cp.asnumpy(X_gpu)

    # # Test different distance metrics
    # for metric in ["l2", "cosine", "dot", "manhattan"]:
    #     # Measure CPU Time (NumPy)
    #     start_cpu = time.time()
    #     top_k_indices_cpu = our_knn_np(N, D, A_cpu, X_cpu, K, distance_metric=metric)
    #     end_cpu = time.time()
    #     cpu_time = end_cpu - start_cpu

    #     # Measure GPU Time (CuPy - Without Kernel Optimization)
    #     start_cp = time.time()
    #     top_k_indices_cp = our_knn(N, D, A_gpu, X_gpu, K, distance_metric=metric, use_kernel=False)
    #     end_cp = time.time()
    #     cp_time = end_cp - start_cp

    #     # Measure GPU Time (CuPy - With Kernel Optimization)
    #     start_elm = time.time()
    #     top_k_indices_element = our_knn(N, D, A_gpu, X_gpu, K, distance_metric=metric, use_kernel=True)
    #     end_elm = time.time()
    #     elm_time = end_elm - start_elm

    #     # Print results
    #     print(f"⚡ {metric.upper()} Distance Results:")
    #     print(f"    CPU Time (NumPy): {cpu_time:.6f} sec")
    #     print(f"    GPU Time (CuPy - No Kernel): {cp_time:.6f} sec")
    #     print(f"    GPU Time (CuPy - Kernel): {elm_time:.6f} sec")
    #     print(f"    Speedup (CPU vs CuPy No Kernel): {round(cpu_time / cp_time, 2)}x")
    #     print(f"    Speedup (CPU vs CuPy Kernel): {round(cpu_time / elm_time, 2)}x")
    #     print(f"    Speedup (CuPy No Kernel vs Kernel): {round(cp_time / elm_time, 2)}x\n")

    ### Test KMeans
    
