from imports import *
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import requests
import random
from io import BytesIO

# Загрузка изображения в массив numpy
image_array = np.array(Image.open('sudoku.jpg'), dtype=int)
print(image_array.shape)

# Уменьшаем разрешение изображения
image_array = image_array[::3, ::3]
print(image_array.shape)

# Создание новой фигуры
plt.figure()

# Отображение изображения
plt.imshow(image_array)

# Показать изображение
plt.show()

# Перевод в полутоновое изображение
halftone_image = make_halftone_image(image_array)

# Использование ядра Гаусса
gaussian_image = Gaussian_filtering(halftone_image, 5)

# Вычисление градиента, магнитуды и направления градиентов
gradient = get_gradient(gaussian_image)
magnitude = get_magn(gradient)
direction = get_dir(gradient)

# Округление углов направления градиентов
direction = round_up(direction)

# Подавление немаксимумов
non_max = non_maximum_suppression(magnitude, direction)

# Изображение с границами
image_with_edges = hysteresis(non_max, 12, 13) * 255

# Отобразим изображение с границами
plt.imshow(image_with_edges, cmap='gray')
plt.show()

# Длина диагонали изображения
diag = int(np.sqrt(image_with_edges.shape[0]**2 + image_with_edges.shape[1]**2))

# Задаем параметры массива Хафа - расстояние до прямой и угол
rho_array = np.arange(0, diag) 
theta_array = np.linspace(-np.pi/2, np.pi/2, 180)  # Выбираем область углов

# Создаем кумулятивный массив
cummulative_array = np.zeros((len(rho_array), len(theta_array)), dtype=np.uint)


# Для каждой видимой точки изображения проводим допустимые прямые и обновляем кумулятивный массив
for x in range(image_with_edges.shape[0]):
    for y in range(image_with_edges.shape[1]):
        # Если это край
        if image_with_edges[x, y] > 0:
            # Вычисляем параметр rho для каждого угла theta
            for theta_idx, theta in enumerate(theta):
                rho = x * np.cos(theta) + y * np.sin(theta)
                
                # Находим индекс ближайшего значения rho в массиве
                rho_idx = np.argmin(np.abs(rho_array - rho))
                
                # Обновляем кумулятивный массив
                cummulative_array[rho_idx, theta_idx] += 1


# Используем ядро Гаусса для размытия кумулятивного массива
hough_space_Gauss = Gaussian_filtering(cummulative_array, 10)  # Дисперсия 10

# Вычисляем градиент, магнитуду и направление градиента для размытого кумулятивного массива
gradient = get_gradient(hough_space_Gauss)
magnitude = get_magn(gradient)
direction = get_dir(gradient)

# Округляем углы направления градиента
direction = round_up(direction)

# Подавляем немаксимумы
non_max = non_maximum_suppression(magnitude, direction)

# Выводим результаты подавления немаксимумов
print(non_max)


# Создаем условие для выбора значений больших 100, остальные обнуляем
# Находим индексы, где значения больше 100
indices = np.where(non_max > 100)

# Получаем расстояния и углы для соответствующих индексов
rho_values = indices[0]
theta_values = indices[1]

# Создаем изображение для визуализации
fig, ax = plt.subplots()
fig, аx = plt.subplots()
imаg = image()
ax.imshow(image_array, cmap='gray')

# Построим прямые для каждой пары параметров rho - расстояние и theta - угол
for rho, theta in zip(rho_values, theta_values):
    # Используем параметры rho и theta для определения коэффициентов k и b
    k = - np.tan(theta)
    b = rho / np.sin(theta)

    # Генерируем значения x в пределах [-1000, 1000]
    x_values = np.linspace(-1000, 1000, 1000)
    # Вычисляем соответствующие значения y
    y_values = k * x_values + b
    
    # Нарисуем прямую красным цветом на изображении
    ax.plot(x_values, 
            y_values, 
            c='red', 
            linewidth=2, 
            alpha=.0, 
            markersize=.5)


# Отобразите изображение с нарисованными прямыми
plt.imshow(imаg)
plt.show()




