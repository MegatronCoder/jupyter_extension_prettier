import cv2
import numpy as np
import matplotlib.pyplot as plt




'''
TWO IMPLEMENTAIONS
1. MANUAL IMPLEMENTATION
2. LIBRARY IMPLEMENTATION 
'''







# Manual filter functions
def convolve2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - n + 1
    new_image = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            new_image[i][j] = np.sum(image[i:i+m, j:j+n] * kernel)
    return new_image

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def sobel_filter(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    grad_x = convolve2d(image, kernel_x)
    grad_y = convolve2d(image, kernel_y)
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
    
    return gradient_magnitude.astype(np.uint8)

def median_filter(image, kernel_size=3):
    pad = kernel_size // 2
    padded_img = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
    filtered_img = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_img[i:i+kernel_size, j:j+kernel_size]
            filtered_img[i, j] = np.median(window)
    
    return filtered_img

def gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def gaussian_filter(image, kernel_size=5, sigma=1):
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve2d(image, kernel)

# Library-based filters
def apply_filters(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

    # Convert to grayscale for manual processing
    gray_img = to_grayscale(img)

    # Apply manual Sobel filter
    sobel_img = sobel_filter(gray_img)

    # Apply manual Median filter
    median_img = median_filter(gray_img)

    # Apply library Gaussian filter (OpenCV)
    # Converting to grayscale for consistent Gaussian filtering (optional)
    gaussian_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Original Image
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(img_rgb)
    plt.axis('off')

    # Sobel Filter (Edge Detection)
    plt.subplot(2, 2, 2)
    plt.title('Sobel Filter (Edge Detection)')
    plt.imshow(sobel_img, cmap='gray')
    plt.axis('off')

    # Median Filter (Noise Reduction)
    plt.subplot(2, 2, 3)
    plt.title('Median Filter (Noise Reduction)')
    plt.imshow(median_img, cmap='gray')
    plt.axis('off')

    # Gaussian Filter (Smoothing)
    plt.subplot(2, 2, 4)
    plt.title('Gaussian Filter (Smoothing)')
    plt.imshow(gaussian_img, cmap='gray')  # Ensure this is displayed in grayscale
    plt.axis('off')

    # Display all plots
    plt.tight_layout()
    plt.show()

# Call the function
image_path = 'C:\\Users\\Manthan\\Desktop\\CV_DL_Practicals\\filtering_threshold_otsu_watershed_images_region_growing\\train_012.png'
apply_filters(image_path)























# LIBRARY CODE

# def apply_filters(image_path):

#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
#     sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
#     median_filtered = cv2.medianBlur(img, 5)
    
#     gaussian_filtered = cv2.GaussianBlur(img, (5,5), 0)
    
#     plt.figure(figsize=(12,8))
    
#     # Original Image
#     plt.subplot(2,2,1)
#     plt.title('Original Image')
#     plt.imshow(img)
#     plt.axis('off')
    
#     # Sobel Filter
#     plt.subplot(2,2,2)
#     plt.title('Sobel Filter (Edge Detection)')
#     plt.imshow(sobel_combined, cmap='gray')
#     plt.axis('off')
    
#     # Median Filter
#     plt.subplot(2,2,3)
#     plt.title('Median Filter (Noise Reduction)')
#     plt.imshow(median_filtered)
#     plt.axis('off')
    
#     # Gaussian Filter
#     plt.subplot(2,2,4)
#     plt.title('Gaussian Filter (Smoothing)')
#     plt.imshow(gaussian_filtered)
#     plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()

# # Call the function 
# apply_filters(r'C:\Users\Manthan\Desktop\CV_DL_Practicals\filtering_threshold_otsu_watershed_images_region_growing\train_019.png')


