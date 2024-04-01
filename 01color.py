'''
    Tarea 6.1 - Color and Filtering
        En esta primera parte, buscamos implementar la determinacion que definen Ohta et al.
'''

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cv2

# Funcion para obtener la matriz de varianza-covarianza
def varianceCovarianceMatrix(img):

	mat = np.zeros((3,3), dtype=float)
	vec = np.zeros((3,1), dtype=float)
	n = np.size(img)

	for i in range(len(img[:,1])):
		for j in range(len(img[1,:])):
			aux_1 = np.array([[img[i,j][0]], [img[i,j][1]], [img[i,j][2]]])
			aux_2 = np.array([img[i,j][0], img[i,j][1], img[i,j][2]])

			mat += aux_1 * aux_2
			vec += aux_1

	C = ((1/n) * (mat)) - ((1/n**2) * (vec * vec.T))

	print(f'La matriz de covarianza es:\n{C}')

	# Obtenemos los valores y vectores propios de C
	W, V = LA.eig(C)

	print(f'\nSus valores propios son: \n{W}\n y sus vectores son: \n{V}\n')

	return C, W, V

if __name__ == "__main__":

	# Leyendo imagen y acomodando los colores
	# img_c = cv2.imread("images/cinves.jpeg")/255
	# img_c = cv2.imread("images/arbol.jpeg")/255
	img_c = cv2.imread("images/franbuesa.jpeg")/255
	# img_c = cv2.imread("images/heart.png")/255

	b, g, r = cv2.split(img_c)
	img_c = cv2.merge([r, g, b])

	# Valores RGB flotantes
	# img_c_norm = np.zeros(img_c.shape, dtype=float)
	# for i in range(len(img_c[:,1])):
	# 	for j in range(len(img_c[1,:])):
	# 		img_c_norm[i,j] = img_c[i,j]/255.0

	# Ontenemos la matriz de covarianza con sus valores y vectores propios
	C, W, V = varianceCovarianceMatrix(img_c)

	# print(img_c_norm.shape[0:2])
	print(img_c[100,200])
	input()

	# Mostrando imagen
	fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7,7))
	ax.imshow(img_c)
	ax.set_title('Color image')
	plt.show()