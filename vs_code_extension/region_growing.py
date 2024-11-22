import numpy as np
from queue import Queue
import cv2
import matplotlib.pyplot as plt

def region_growing(image, seed_point, threshold):
    
    # Ensure input image is grayscale
    if len(image.shape) > 2:
        raise ValueError("Input image must be grayscale (2D array)")
    
    # Create a mask to store the segmented region
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Get image dimensions
    height, width = image.shape
    print(image.shape)
    
    # Get the seed point intensity
    seed_intensity = image[seed_point]
    
    # Initialize queue for region growing
    queue = Queue()
    queue.put(seed_point)
    
    # Define 8-connectivity neighborhood
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    # Mark the seed point in the mask
    mask[seed_point] = 1
    
    # Process pixels in the queue
    while not queue.empty():
        current_row, current_col = queue.get()
        
        # Check all neighboring pixels
        for dy, dx in neighbors:
            new_row, new_col = current_row + dy, current_col + dx
            
            # Check if the neighbor is within image boundaries
            if (0 <= new_row < height and 
                0 <= new_col < width and 
                mask[new_row, new_col] == 0):
                
                # Calculate intensity difference
                intensity_diff = abs(float(image[new_row, new_col]) - float(seed_intensity))
                
                # If the difference is within threshold, add to region
                if intensity_diff <= threshold:
                    mask[new_row, new_col] = 1
                    queue.put((new_row, new_col))
    
    return mask

def multiple_seed_region_growing(image, seed_points, threshold):

    final_mask = np.zeros_like(image, dtype=np.uint8)
    
    for seed_point in seed_points:
        mask = region_growing(image, seed_point, threshold)
        final_mask = np.logical_or(final_mask, mask)
    
    return final_mask.astype(np.uint8)

# Usage example
if __name__ == "__main__":

    # Load image
    image = cv2.imread(r"D:\College\pccoe\5th sem\CV\Practicals\Final_practicals\something-main\something-main\dog\Dog Segmentation\Images\dog.8790.jpg", cv2.IMREAD_GRAYSCALE)
    bilateral = cv2.bilateralFilter(image, 5, 15, 15) 

    # Define seed points
    seed_points = [(179, 250) , (100,50), (350,450)]

    # Set threshold
    threshold = 46.5

    # Apply region growing
    segmented = multiple_seed_region_growing(bilateral, seed_points, threshold)

    # Display result
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("Original Image")
    plt.imshow(bilateral, cmap="gray")
    plt.subplot(122)
    plt.title("Segmented Image")
    plt.imshow(segmented, cmap="gray")
    plt.show()
