import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import os


@jit
def calc_julia(c, maxiter, *args):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z**2 + -0.8+0.156j

    return maxiter


@jit
def calc_mandelbrot(c, maxiter):
    z = c
    for n in range(maxiter):
        if np.abs(z) > 2:
            return n
        z = z*z + c
    return maxiter


@jit
def calc_burning(c, maxiter):
    z = c
    for n in range(maxiter):
        if np.abs(z) > 2:
            return n
        z = (abs(np.real(z)) + 1j * abs(np.imag(z))) ** 2 + c
    return maxiter


@jit
def calc_etfractal(xmin, xmax, ymin, ymax, res, maxiter, ftype):
    if ftype == 'burning':
        calculate = calc_burning
    elif ftype == 'mandelbrot':
        calculate = calc_mandelbrot
    elif ftype == "julia":
        calculate = calc_julia
    else:
        print("invalid fractal name")
        return 0

    x = np.linspace(xmin, xmax, res)    #[xmin + x * (xmax - xmin) / res for x in range(res)]
    y = np.linspace(ymax, ymin, res)    #[ymax + y * (ymax - ymin) / res for y in range(res)]
    n3 = np.empty((res, res))
    print("calculating mandelbrot set...")
    for xcol in range(res):
        for ycol in range(res):
            n = calculate(x[xcol] + 1j * y[ycol], maxiter)
            n3[ycol, xcol] = n

    print("mandelbrot set calculated\nplotting...")

    if normalise:
        n3[1, 1] = 0
        n3[0, 0] = maxiter
    return n3


x = 0
y = 0
def onclick(event):
    global side
    global ftype
    global zoom
    global maxiter
    global x, y
    global additer

    if event.key == 'u':
        print("zooming out")
        side = side * zoom
        maxiter -= additer

    elif event.key == 's':
        print('saving image')
        filenew = False
        i = 1
        while not filenew:
            if os.path.isfile(ftype + str(i) + ".png"):
                i += 1
            else:
                filenew = True
        plt.savefig(ftype + str(i) + ".png", bbox_inches='tight', dpi=500)

    elif event.button:
        side /= zoom
        x = event.xdata
        y = event.ydata
        print("zooming to point", (x, y))
        maxiter += additer
    else:
        print("something went wrong.")
        return 0

    xmin, xmax, ymin, ymax = x - side, x + side, y - side, y + side
    graph = calc_etfractal(xmin, xmax, ymin, ymax, res, maxiter, ftype)
    # clear frame
    plt.clf()
    plt.axis('off')
    plt.imshow(graph, extent=[xmin, xmax, ymin, ymax], cmap=cmap, interpolation='none')
    plt.draw()


fig, ax = plt.subplots()
res = 2000
maxiter = 100
additer = 0
cmap = "twilight"
ftype = 'burning'
side = 2
zoom = 10
normalise = False

n3 = calc_etfractal(-side, side, -side, side, res, maxiter, ftype)
ax.imshow(n3, extent=[-side, side, -side, side], cmap=cmap, interpolation='none')
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onclick)

plt.show()
plt.draw()
