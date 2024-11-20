import cv2
import numpy as np
import matplotlib.pyplot as plt





# MANUAL IMPLEMENTAION

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

def sobel_filter(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    grad_x = convolve2d(image, kernel_x)
    grad_y = convolve2d(image, kernel_y)
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
    
    return gradient_magnitude.astype(np.uint8)

def display_image(image, title="Image"):
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def process_image(image_path):
    image = cv2.imread(image_path)
    gray_image = to_grayscale(image)
    
    sobel_image = sobel_filter(gray_image)
    display_image(gray_image, "Gray Image")
    display_image(sobel_image, "Sobel Filter")

process_image(r'C:\Users\Manthan\Desktop\CV_DL_Practicals\edge_images\7d35e395-ae41f911.jpg')






# LIBRARY IMPLEMENTAION


# def process_image(image_path):
#     # implement sobel filter by library
#     img = cv2.imread(image_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
#     sobel_x = cv2.convertScaleAbs(sobel_x)
#     sobel_y = cv2.convertScaleAbs(sobel_y)
#     sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.title('Original Image')
#     plt.imshow(img_rgb)
#     plt.axis('off')
#     plt.subplot(1, 2, 2)
#     plt.title('Sobel Filter')
#     plt.imshow(sobel, cmap='gray')
#     plt.axis('off')
#     plt.show()
