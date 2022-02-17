import numpy as np
from numpy.linalg import solve
from matplotlib import cm
import matplotlib.pyplot as plt

class Puasson:
    def __init__(self, x1, xn, y1, yn, n, m, sigma, g1, g2, g3, g4):
        self.x1 = x1
        self.xn = xn
        self.y1 = y1
        self.yn = yn
        self.n = n
        self.m = m
        self.sigma = sigma
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.g4 = g4

    def transformation(self, i, j, n):
        return j * n + i

    def border(self, A, B, n, m, x, y):
        k3 = 0
        for j in range(0, m):
            for i in range(0, n):
                k3 = self.transformation(i, j, n)
                if i == 0:
                    A[k3][k3] = 1
                    code = compile(g1, "<string>", "eval")
                    B[k3] = eval(code, {"np": np, "x": x[i], "y": y[j]})
                if i == n - 1:
                    A[k3][k3] = 1
                    code = compile(g2, "<string>", "eval")
                    B[k3] = eval(code, {"np": np, "x": x[i], "y": y[j]})
                if j == 0:
                    A[k3][k3] = 1
                    code = compile(g3, "<string>", "eval")
                    B[k3] = eval(code, {"np": np, "x": x[i], "y": y[j]})
                if j == m - 1:
                    A[k3][k3] = 1
                    code = compile(g4, "<string>", "eval")
                    B[k3] = eval(code, {"np": np, "x": x[i], "y": y[j]})
        return A, B


    def inside(self, A, B, n, m, x, y, dx, dy, sigma):
        k1 = k2 = k3 = k4 = k5 = 0
        for j in range(1, m - 1):
            for i in range(1, n - 1):
                k1 = self.transformation(i - 1, j, n)
                k2 = self.transformation(i + 1, j, n)
                k3 = self.transformation(i, j, n)
                k4 = self.transformation(i, j - 1, n)
                k5 = self.transformation(i, j + 1, n)
                A[k3][k1] = 1 / (dx ** 2)
                A[k3][k2] = 1 / (dx ** 2)
                A[k3][k3] = -2 / (dx ** 2) - 2 / (dy ** 2)
                A[k3][k4] = 1 / (dy ** 2)
                A[k3][k5] = 1 / (dy ** 2)
                code = compile(sigma, "<string>", "eval")
                B[k3] = eval(code, {"np": np, "x": x[i], "y": y[j]}) / (8.85418781762039 * (10 ** (-12)))
        return A, B

    def convert(self):
        matrix_A = [[0 for j in range(0, self.n * self.m)] for i in range(0, self.n * self.m)]
        matrix_B = [0 for i in range(0, self.m * self.n)]

        dx = (self.xn - self.x1) / (self.n - 1)
        dy = (self.yn - self.y1) / (self.m - 1)

        mas_x = [i * dx + self.x1 for i in range(0, self.n)]
        mas_y = [i * dy + self.y1 for i in range(0, self.m)]

        A, B = self.inside(matrix_A, matrix_B, self.n, self.m, mas_x, mas_y, dx, dy, self.sigma)
        A, B = self.border(A, B, self.n, self.m, mas_x, mas_y)
        U = solve(A, B)

        Uij = []
        k = 0
        for i in range(self.n):
            Uij.append([])
            for j in range(self.m):
                Uij[i].append(U[k])
                k += 1
        return U, mas_x, mas_y

    def plotting(self):
        U, mas_x, mas_y = self.convert()
        plt.figure()
        ax = plt.axes(projection='3d')
        from mpl_toolkits.mplot3d import Axes3D
        X, Y = np.meshgrid(mas_x, mas_y)
        nu = 1 - X ** 2 - Y ** 2

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, nu)


        for i in range(len(mas_y)):
            X = np.array(mas_x)
            Y = np.array([mas_y[i] for j in range(len(mas_y))])
            Z = np.array(U[0 + i * len(mas_y):len(mas_y) + len(mas_y) * i])
            # ax.plot3D(X, Y, Z, 'green')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

#
x1 = float(input("Введите значение x1: "))
xn = float(input("Введите значение xn: "))
y1 = float(input("Введите значение y1: "))
yn = float(input("Введите значение yn: "))
n = int(input("Введите значение n: "))
m = int(input("Введите значение m: "))
sigma = str(input("Введите функцию правой части уравнения Пуассона: "))
g1 = str(input("Введите функцию g1 на левой границе: "))
g2 = str(input("Введите функцию g2 на правой границе : "))
g3 = str(input("Введите функцию g3 на верхней границе: "))
g4 = str(input("Введите функцию g4 на нижней границе: "))

model = Puasson(x1, xn, y1, yn, n, m, sigma, g1, g2, g3, g4)
model.plotting()