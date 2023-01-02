import numpy as np
from PIL import Image

# Open image, convert to black and white mode
image = Image.open('images/resultado_extract.jpg').convert('1')
# image = Image.open('rIlXS.jpg').convert('1')
# image = Image.open('4x14x187.bmp').convert('1')


w, h = image.size

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


array = (np.array(image.resize((int(w / wt), int(h / ht))))).astype(int)
print(array.tolist())
print("\n\n\n")

print(array.shape)
