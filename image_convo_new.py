import numpy as np
from skimage import feature

# Load the images and labels
images = []
labels = []
for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features from the image
    hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    
    # Add the features and label to the list
    images.append(hog_features)
    labels.append(label)

# Convert the lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Save the arrays to .npy files
np.save('hog_features.npy', images)
np.save('labels.npy', labels)
