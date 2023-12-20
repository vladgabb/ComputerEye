import numpy as np
from PIL import Image
from pylab import *
import random as rd

np.seterr(divide='ignore', invalid='ignore')

image = np.array(Image.open('BMW1.jpg'), dtype=int)
figure()
imshow(image)
show()

image2 = np.array(Image.open('BMW2.jpg'), dtype=int)
figure()
imshow(image2)
show()

image3 = np.array(Image.open('BMW3.jpg'), dtype=int)
figure()
imshow(image3)
show()


def make_halftone_image(image):
    # Получаем ширину и высоту исходного изображения
    width = image.shape[0]
    height = image.shape[1]

    # Создаем пустой массив для полутонового изображения
    halftoneImage = np.zeros(width * height, dtype=int)
    halftoneImage.shape = (width, height)

    # Проходим по каждому пикселю исходного изображения
    for i in range(width):
        for j in range(height):
            # Вычисляем среднее значение по каналам RGB для текущего пикселя
            halftoneImage[i][j] = np.mean(image[i, j, :])

    return halftoneImage

halftoneImage = make_halftone_image(image)
figure()
imshow(halftoneImage, cmap='gray', vmin=0, vmax=255)
show()

halftoneImage2 = make_halftone_image(image2)

halftoneImage3 = make_halftone_image(image3)


def build_G(sigma2):
    # Вычисление констант для фильтра Гаусса
    const1 = 2 * sigma2
    const2 = 2 * np.pi * sigma2

    # Создание пустой матрицы G размером 3x3 для хранения фильтра Гаусса
    G = np.zeros(9, dtype=np.float64)
    G.shape = (3, 3)

    # Заполнение матрицы G значениями фильтра Гаусса
    for i in range(3):
        for j in range(3):
            G[i, j] = np.power(np.e, -((i - 3 // 2) ** 2 + (j - 3 // 2) ** 2) / const1) / const2

    # Нормализация фильтра Гаусса, чтобы сумма его значений была равна 1
    G /= np.sum(G)
    return G

def Gaussian_filtering(image, sigma2):
    # Строим фильтр Гаусса
    G = build_G(sigma2)

    width = image.shape[0]
    height = image.shape[1]

    # Создаем пустое изображение для результата
    gaussImage = np.zeros(width * height, dtype=int)
    gaussImage.shape = (width, height)

    # Создаем копию изображения, дополненную нулями по краям
    imageCopy = np.zeros((width + 1) * (height + 1), dtype=int)
    imageCopy.shape = ((width + 1), (height + 1))

    for i in range(width):
        for j in range(height):
            imageCopy[i + 1, j + 1] = image[i, j]

    # Применяем фильтр Гаусса к изображению
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            gaussImage[i, j] = np.sum(G * imageCopy[i - 1:i + 2, j - 1:j + 2])

    # Обрезаем значения, чтобы они находились в диапазоне 0-255
    gaussImage = np.clip(gaussImage, 0, 255)

    return gaussImage


gaussImage = Gaussian_filtering(halftoneImage, 100)
figure()
imshow(gaussImage, cmap='gray', vmin=0, vmax=255)
show()

gaussImage2 = Gaussian_filtering(halftoneImage2, 100)

gaussImage3 = Gaussian_filtering(halftoneImage3, 100)

def get_gradient(image):
    # Получаем ширину и высоту изображения
    width = image.shape[0]
    height = image.shape[1]

    # Создаем копию изображения, дополненную нулями по краям
    imageCopy = np.zeros((width + 1, height + 1), dtype=int)

    for i in range(width):
        for j in range(height):
            imageCopy[i + 1, j + 1] = image[i, j]

    # Создаем массив gradient для хранения градиентов
    gradient = np.zeros((width + 1, height + 1, 2), dtype=float)

    # Вычисляем градиенты для каждого пикселя внутри изображения
    for i in range(1, width):
        for j in range(1, height):
            # Вычисляем горизонтальную компоненту градиента (dx)
            gradient[i, j, 0] = (2 * imageCopy[i + 1, j] - 2 * imageCopy[i - 1, j] + imageCopy[i + 1, j - 1] - imageCopy[i - 1, j - 1] + imageCopy[i + 1, j + 1] - imageCopy[i - 1, j + 1]) / 6

            # Вычисляем вертикальную компоненту градиента (dy)
            gradient[i, j, 1] = (2 * imageCopy[i, j + 1] - 2 * imageCopy[i, j - 1] + imageCopy[i - 1, j + 1] - imageCopy[i - 1, j - 1] + imageCopy[i + 1, j + 1] - imageCopy[i + 1, j - 1]) / 6

    # Возвращаем градиенты, обрезая нулевые значения по краям
    return gradient[1:width + 1, 1:height + 1, :]


gradien = get_gradient(gaussImage)
gradient2 = get_gradient(gaussImage2)
gradient3 = get_gradient(gaussImage3)
print(f'Gradient: {gradien}')


def get_magn(gradient):
    # Получаем ширину и высоту градиентного изображения
    width = gradient.shape[0]
    height = gradient.shape[1]

    # Создаем пустой массив `magn` для хранения магнитуды градиента
    magn = np.zeros((width, height), dtype=float)

    # Вычисляем магнитуду для каждого пикселя в градиентном изображении
    for i in range(width):
        for j in range(height):
            # Используем формулу для вычисления магнитуды градиента
            magn[i, j] = np.sqrt(gradient[i, j, 0]**2 + gradient[i, j, 1]**2)

    return magn


magn = get_magn(gradien)
magn2 = get_magn(gradient2)
magn3 = get_magn(gradient3)
print(f'Magnituda: {magn}')


def get_dir(gradient):
    # Получаем ширину и высоту градиентного изображения
    width = gradient.shape[0]
    height = gradient.shape[1]
  
    # Создаем пустой массив `dir` для хранения направления градиента
    dir = np.zeros((width, height), dtype=float)

    for i in range(width):
        for j in range(height):
            if gradient[i, j, 1] == 0:
                # Если вертикальная компонента равна 0, устанавливаем направление в 0 (горизонтальное)
                dir[i, j] = 0
            elif gradient[i, j, 0] == 0:
                # Если горизонтальная компонента равна 0, устанавливаем направление в pi/2 (вертикальное)
                dir[i, j] = np.pi / 2
            else:
                # Иначе, вычисляем арктангенс от отношения вертикальной к горизонтальной компоненте градиента
                dir[i, j] = np.arctan(gradient[i, j, 1] / gradient[i, j, 0])

    return dir


dir = get_dir(gradien)
dir2 = get_dir(gradient2)
dir3 = get_dir(gradient3)
print(f'Direction: {dir}')

def round_up(dir):
    # Создаем новый массив `new_dir` с целочисленным типом данных и теми же размерами, что и `dir`
    new_dir = np.zeros_like(dir, dtype=int)

    # Преобразуем углы в градусах в диапазоне [0, 360) и округляем
    dir = (dir * 180 / np.pi) % 360

    for i in range(dir.shape[0]):
        for j in range(dir.shape[1]):
            if dir[i, j] != 0:
                # Рассчитываем, в какой из 8 категорий попадает угол и округляем его
                a = dir[i, j] // 45
                b = (dir[i, j] % 45) // 22.5
                new_dir[i, j] = (a + b) * 45 % 360

    return new_dir

round_up_dir = round_up(dir)
round_up_dir2 = round_up(dir2)
round_up_dir3 = round_up(dir3)
print(round_up_dir)


def non_maximum_suppression(magn, dir):
    # Получаем ширину и высоту магнитуды градиента
    w = magn.shape[0]
    h = magn.shape[1]

    # Находим максимальное значение магнитуды
    max = magn.max()

    # Создаем копию магнитуды с дополнительной границей нулей
    magnCopy = np.ones((w + 2, h + 2), dtype=float)
    magnCopy *= max

    for i in range(1, w + 1):
        for j in range(1, h + 1):
            magnCopy[i, j] = magn[i - 1, j - 1]

    # Создаем копию магнитуды для немаксимумов
    non_max = magn.copy()

    for i in range(1, w + 1):
        for j in range(1, h + 1):
            for k, x, y in [[0, 0, 1], [45, 1, 1], [90, 1, 0], [135, -1, 1]]:
                if dir[i - 1, j - 1] % 180 == k:
                    if magnCopy[i, j] <= magnCopy[i + x, j + y] and magnCopy[i, j] <= magnCopy[i - x, j - y]:
                        non_max[i - 1, j - 1] = 0

    return non_max


non_max = non_maximum_suppression(magn, round_up_dir)
non_max2 = non_maximum_suppression(magn2, round_up_dir2)
non_max3 = non_maximum_suppression(magn3, round_up_dir3)
print(non_max.min(), non_max.max())
print(non_max)


def hysteresis(magn, min, max):
    # Получаем ширину и высоту магнитуды градиента
    w = magn.shape[0]
    h = magn.shape[1]

    # Создаем копию магнитуды с дополнительной границей нулей
    magnCopy = np.zeros((w + 2, h + 2), dtype=float)
    for i in range(1, w + 1):
        for j in range(1, h + 1):
            magnCopy[i, j] = magn[i - 1, j - 1]

    # Создаем массив `hyst` для хранения состояния пикселей (0 - фон, 1 - неопределен, 2 - граница)
    hyst = np.ones((w + 2, h + 2), dtype=int) * 3
    hyst[1:w + 1, 1:h + 1] -= 4

    next = []

    def cycle(i, j):
        for x, y in (np.argwhere(hyst[i - 1:i + 2, j - 1:j + 2] <= 0) - 1):
            if magnCopy[i + x, j + y] > min:
                hyst[i + x, j + y] = 2
                next.append([i + x, j + y])
            else:
                hyst[i + x, j + y] = 1

    i, j = np.argwhere(magnCopy[:, :] > max)[0]
    hyst[i, j] = 2
    cycle(i, j)

    while hyst.min() == -1:
        if len(next) > 0:
            for i, j in next:
                if hyst[i, j] == -1:
                    if magnCopy[i, j] > max:
                        hyst[i, j] = 2
                        cycle(i, j)
                    else:
                        hyst[i, j] = 0
            next.clear()
        else:
            i, j = np.argwhere(hyst[:, :] == -1)[0]
            if magnCopy[i, j] > max:
                hyst[i, j] = 2
                cycle(i, j)
            else:
                hyst[i, j] = 0

    return hyst[1:w + 1, 1:h + 1] // 2


hyst = hysteresis(non_max, 10, 15) * 255
figure()
imshow(hyst, cmap='gray', vmin=0, vmax=255)
show()

hyst2 = hysteresis(non_max2, 10, 15) * 255
figure()
imshow(hyst2, cmap='gray', vmin=0, vmax=255)
show()

hyst3 = hysteresis(non_max3, 10, 15) * 255
figure()
imshow(hyst3, cmap='gray', vmin=0, vmax=255)
show()

print(0)