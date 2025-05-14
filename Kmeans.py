import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def kMeans_init_centroids(X, K):
   
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids

def compute_centroids(X, idx, K):

        # Useful variables
    m, n = X.shape
    
    # Initialize centroids array
    centroids = np.zeros((K, n))
    
    ### START CODE HERE ###
    for i in range(K):
        # Find the indices of all points assigned to centroid `i`
        points_in_cluster = X[np.where(idx == i)]
        
        if points_in_cluster.shape[0] == 0:  # No points assigned to this centroid
            # If no points are assigned, keep the centroid as is (or handle appropriately)
            continue
        
        # Compute the mean of points assigned to the centroid `i`
        centroids[i] = np.mean(points_in_cluster, axis=0)
    ### END CODE HERE ###
    
    return centroids

# UNQ_C1
# GRADED FUNCTION: find_closest_centroids

def find_closest_centroids(X, centroids):
   

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###
    
    for _ in range(100):
        y=[]

        for data_point in X:
            distances=np.sqrt(np.sum((centroids - data_point)**2 ,axis=1))
            cluster_num=np.argmin(distances)
            y.append(cluster_num)
        y=np.array(y)
        
      

    for i in range(K):
         idx[np.argwhere(y == i)] = i 
      
            
        
     ### END CODE HERE ###
    
    return idx




def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
   
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # Optionally plot progress
        # if plot_progress:
        #     plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
        #     previous_centroids = centroids
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx





original_img = plt.imread('bird_small1.png')
plt.imshow(original_img)
print("Shape of original_img is:", original_img.shape)

original_img_rgb = original_img[:, :, :3]
print()
X_img = np.reshape(original_img_rgb, (original_img_rgb.shape[0] * original_img_rgb.shape[1], 3))


K = 16
max_iters = 10

# Using the function you have implemented above. 
initial_centroids = kMeans_init_centroids(X_img, K)

# Run K-Means - this can take a couple of minutes depending on K and max_iters
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)

#plot_kMeans_RGB(X_img, centroids, idx, K)
#show_centroid_colors(centroids)

idx = find_closest_centroids(X_img, centroids)

# Replace each pixel with the color of the closest centroid
X_recovered = centroids[idx, :] 

# Reshape image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img_rgb.shape) 




# Display original image
fig, ax = plt.subplots(1,2, figsize=(16,16))
plt.axis('off')

ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()


# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()