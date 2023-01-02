import cv2
import numpy as np

# Open image, convert to black and white mode
# image = Image.open('resultado_extract.jpg').convert('1')
# image = Image.open('rIlXS.jpg').convert('1')
# image = Image.open('4x14x187.bmp').convert('1')

image = cv2.imread('resultado_extract.jpg', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('resultado_extract_preenchido.jpg', cv2.IMREAD_GRAYSCALE)

# w, h = image.size
w, h = image.shape

# Temporary NumPy array of type bool to work on
temp = np.array(image)

# Detect changes between neighbouring pixels
diff_y = np.diff(temp, axis=0)
diff_x = np.diff(temp, axis=1)

# Create union image of detected changes
temp = np.zeros_like(temp)
temp[:h - 1, :] |= diff_y
temp[:, :w - 1] |= diff_x

# Calculate distances between detected changes
diff_y = np.diff(np.nonzero(np.diff(np.sum(temp, axis=0))))
diff_x = np.diff(np.nonzero(np.diff(np.sum(temp, axis=1))))

# Calculate tile height and width
ht = np.median(diff_y[diff_y > 1]) + 38
wt = np.median(diff_x[diff_x > 1]) + 38

# Resize image w.r.t. tile height and width


dim = (int(w / wt), int(h / ht))

img_resize = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

thresh = 127
im_bw = cv2.threshold(img_resize, thresh, 255, cv2.THRESH_BINARY)[1]

array = (np.array(im_bw))
print(array)
print("\n\n\n")

print(array.shape)
