import os
import cv2
import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples

# --- Configuration ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(SCRIPT_DIR, 'image_dataset/')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, 'greenery_model_dynamic.pkl')
SAMPLE_SIZE_FOR_K = 30  # Number of images to sample to find optimal 'k'
K_RANGE = range(2, 8)   # Test clusters from 2 to 7

def get_pixel_sample(folder_path, sample_limit, img_size=(100, 100)):
    """Loads a small sample of data to analyze color distributions."""
    pixels = []
    valid_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in valid_files[:sample_limit]:
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Crucial: Match colors to inference script
            img = cv2.resize(img, img_size)
            pixels.append(img.reshape((-1, 3)))
            
    return np.vstack(pixels) if pixels else np.array([])

def find_dynamic_k(pixels, k_range):
    """Dynamically finds the optimal k and plots the visual proof."""
    print("Analyzing color distribution to find optimal k...")
    inertias = []
    
    for k in k_range:
        mbk = MiniBatchKMeans(n_clusters=k, batch_size=2048, n_init='auto', random_state=42)
        mbk.fit(pixels)
        inertias.append(mbk.inertia_)

    # 1. The Math (Finding the Elbow)
    p1 = np.array([k_range[0], inertias[0]])
    p2 = np.array([k_range[-1], inertias[-1]])
    
    distances = []
    for i, k in enumerate(k_range):
        p3 = np.array([k, inertias[i]])
        distance = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        distances.append(distance)
        
    optimal_k = k_range[np.argmax(distances)]
    print(f"-> Optimal clusters (k) determined as: {optimal_k}")

    # 2. The Visualization (Plotting the Proof)
    plt.figure(figsize=(10, 6))
    
    # Plot the actual inertia curve
    plt.plot(k_range, inertias, marker='o', label='Inertia Curve (The Arm)')
    
    # Plot the straight line from start to end
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', label='Reference Line (Shoulder to Wrist)')
    
    # Highlight the chosen optimal k
    optimal_index = list(k_range).index(optimal_k)
    plt.plot(optimal_k, inertias[optimal_index], 'g*', markersize=15, label=f'Optimal k = {optimal_k} (The Elbow)')
    
    # Draw a line showing the actual perpendicular distance
    plt.vlines(x=optimal_k, ymin=inertias[optimal_index], ymax=p1[1] - ((p1[1]-p2[1])/(p1[0]-p2[0])) * (p1[0]-optimal_k), 
               colors='green', linestyles='dotted', label='Max Distance')

    plt.title('Programmatic Elbow Method Optimization')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Error)')
    plt.legend()
    plt.grid(True)
    
    # This will pause the script and show the graph until you close the window
    plt.show() 

    return optimal_k

def train_final_model(folder_path, k):
    """Trains the model on the full dataset using partial_fit for memory efficiency."""
    print(f"Starting full dataset training with k={k}...")
    final_model = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init='auto', random_state=42)
    
    valid_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    processed_count = 0
    
    for filename in valid_files:
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (150, 150))
            pixels = img.reshape((-1, 3))
            
            final_model.partial_fit(pixels)
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images...")

    return final_model

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Grab a representative sample of pixels
    sample_data = get_pixel_sample(TRAIN_DIR, SAMPLE_SIZE_FOR_K)
    
    if len(sample_data) == 0:
        print("Error: No images found in the training directory.")
    else:
        # 2. Let the algorithm decide the best 'k'
        optimal_k = find_dynamic_k(sample_data, K_RANGE)
        
        # 3. Train on everything
        trained_model = train_final_model(TRAIN_DIR, optimal_k)
        
        
        # 4. Save the "brain"
        joblib.dump(trained_model, MODEL_SAVE_PATH)
        print(f"Success! Model saved to {MODEL_SAVE_PATH}")

        # Calculate silhouette score on sample data
        print("\nCalculating silhouette score...")
        silhouette_avg = silhouette_score(sample_data, trained_model.predict(sample_data))
        print(f"Silhouette Score: {silhouette_avg:.4f}")

        # After training, generate silhouette plot for cluster quality diagnostics
        labels = trained_model.predict(sample_data)
        silhouette_vals = silhouette_samples(sample_data, labels)
        silhouette_avg = silhouette_score(sample_data, labels)

        fig, ax = plt.subplots(figsize=(10, 6))
        y_lower = 10

        for i in range(optimal_k):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
    
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
    
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_vals,
                alpha=0.7,
                label=f'Cluster {i}'
            )
            y_lower = y_upper + 10

        ax.axvline(x=silhouette_avg, color="red", linestyle="--", label=f'Avg Score: {silhouette_avg:.3f}')
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster Label")
        ax.set_title("Silhouette Plot for Each Cluster")
        ax.legend()
        plt.tight_layout()
        plt.show()
        fig, axes = plt.subplots(1, optimal_k, figsize=(12, 3))
        centers = trained_model.cluster_centers_
        
        for i in range(optimal_k):
            color_square = np.ones((100, 100, 3), dtype=np.uint8)
            color_square[:, :] = centers[i].astype(np.uint8)
            axes[i].imshow(color_square)
            axes[i].set_title(f'Cluster {i} Color')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()