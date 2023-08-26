from PIL import Image
import numpy as np


def convNet(mat, kernel):
    r = mat.shape[0]
    k = kernel.shape[0]
    res = []
    i = 0
    while i < r-(k-1):
        tmpl = []
        j = 0
        while j < r-(k-1):
            x = sum(sum(mat[i:i+k, j:j+k]*kernel))
            tmpl.append(x)
            j += 1
        res.append(tmpl)
        i += 1
    return res


m = np.array([[1, 2, 3, 4, 5], [1, 3, 2, 3, 10], [
    3, 2, 1, 4, 5], [6, 1, 1, 2, 2], [3, 2, 1, 5, 4]])
k = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
print(convNet(m, k))


im = Image.open('ex1.jpg')
rgb = np.array(im.convert('RGB'))
r = rgb[:, :, 0]  # array of R pixels

kernel1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel2 = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]])

# # Image.fromarray(np.uint8(r)).show()
# Image.fromarray(np.uint8(convNet(r, kernel1))).show()
Image.fromarray(np.uint8(convNet(r, kernel2))).show()
