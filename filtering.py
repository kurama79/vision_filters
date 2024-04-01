'''
    Tarea 6.2 - Color and Filtering
        En esta segunda parte, haremos una funcion que implemente un filtro Gaussiano.
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Funcion para aplicar filtro Gaussiano
def gaussinaFilter(img, sigma, gaus_cut):

	filter_size = int(gaus_cut * sigma)
	height = img.shape[0] - filter_size + 1
	width = img.shape[1] - filter_size + 1

	img2filter = np.zeros((height*width, filter_size**2))
	
	count = 0
	for i in range(height):
		for j in range(width):
			aux = np.ravel(img[i:i+filter_size, j:j+filter_size])
			img2filter[count, :] = aux

			count += 1

	# Obtenemos el Kernel
	kernel_1 = gaussianKernel(filter_size, sigma)
	kernel_2 = gaussianLaplaceKernel(filter_size, sigma)

	filter4img_1 = np.ravel(kernel_1)
	filter4img_2 = np.ravel(kernel_2)

	# Aplicnado el filtro
	filtered_img_1 = np.dot(img2filter, filter4img_1).reshape(height, width).astype('uint8')
	filtered_img_2 = np.dot(img2filter, filter4img_2).reshape(height, width).astype('uint8')

	return filtered_img_1, filtered_img_2

# Funcion para obtener el Kernel del filtro Gaussiano
def gaussianKernel(size, sigma):
	c = int(size / 2)
	x, y = np.mgrid[0-c:size-c, 0-c:size-c]

	# Funcion de Gauss
	kernel = (1 / (np.sqrt(2*np.pi)*sigma)) * np.exp(-(np.square(x) + np.square(y)) / (2*sigma**2))

	return kernel

# Funcion para obtener el Kernel del filtro Gaussiano-Laplaciano
def gaussianLaplaceKernel(size, sigma):
	c = int(size / 2)
	x, y = np.mgrid[0-c:size-c, 0-c:size-c]

	# Laplaciano de Gauss
	kernel = ((np.square(x) + np.square(y) - 2*(sigma**2)) / (np.pi*sigma**4)) * np.exp(-(np.square(x) + np.square(y))/(2*(sigma**2)))

	return kernel


if __name__ == "__main__":

	# Leyendo la imagen
	img = cv2.imread('images/santaMa1.jpeg', cv2.IMREAD_GRAYSCALE).astype(float)

	# Para el padding
	scale = 1
	width = int(img.shape[1] * scale)
	height = int(img.shape[0] * scale)

	# Resize 
	img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

	img_output_gauss, img_output_laplace = gaussinaFilter(img, 1, 4)
	img_output_gauss_1, img_output_laplace_1 = gaussinaFilter(img_output_gauss, 1, 4)
	# img_output_gauss_2, img_output_laplace_2 = gaussinaFilter(img_output_gauss_1, 1, 4)

	# Mostrando imagenes orgianl y filtro
	plt.gray()
	plt.subplot(131), plt.imshow(img), plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(132), plt.imshow(img_output_gauss_1), plt.title('Gaussian')
	plt.xticks([]), plt.yticks([])
	plt.subplot(133), plt.imshow(img_output_laplace), plt.title('Laplace-Gaussian')
	plt.xticks([]), plt.yticks([])
	plt.show()