import cv2
import numpy as np
import argparse
import glob

#realiza a leitura da imagem
image = cv2.imread('test_image.jpg')
#copia a imagem para dentro de outra variavel
lane_image = np.copy(image)
#converte a imagem para tons de cinza 0 a 255
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
#aplica o blur na imagem para remocao de ruidos
blur = cv2.GaussianBlur(gray, (5, 5), 0)

#funcao que retorna imagem com canny aplicado
def auto_canny(imagem, sigma=0.33):
	# mediana do gradiente da imagem
	v = np.median(imagem)
	#realiza o calculo dos thresholds a partir da mediana
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(imagem, lower, upper)
	# return da funcao
	return edged
#chama a funcao de bordas utilizando a imagem com blur
canny = auto_canny(blur)
cv2.imshow('Resultado', canny)
cv2.waitKey(0)
