from numba import jit
import numpy as np
from matplotlib import pyplot as plt
import time


#class fractal:
#def __init__(self, ftype, iteration, savefig=False, z0 = 3):
#    z0 = 3.001
#
#    self.ftype = ftype
#    self.savefig = savefig
#    self.iteration = iteration
#
#    escape_time(z0, ftype)

@jit
def calc_mandelbrot(c, maxiter, z0):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z**2 + c
    return maxiter

@jit
def calc_julia(c, maxiter, z0):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z**2 + z0

    return maxiter


@jit
def calc_burning(c, maxiter, z0):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = (abs(z.real) + 1j*abs(z.imag)) * (abs(z.real) + 1j*abs(z.imag)) + c
    return maxiter


@jit
def calc_genJulia(c, maxiter, z0):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        if z.imag*z.real == 0:
            return 0
        coz = np.conj(z)
        z = (z.real/z.imag)**2 + coz*1j
    return maxiter

@jit
def calc_mandelbar(c, maxiter, z0):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        conz = np.conj(z)
        z = conz ** 5 + c
    return maxiter


@jit
def calc_etfractal(xmin, xmax, ymin, ymax, res, maxiter, z0, ftype):
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymax, ymin, res)
    n3 = np.empty((res, res))

    if ftype == 'mandelbrot':
        calculate = calc_mandelbrot
    elif ftype == 'julia':
        calculate = calc_julia
    elif ftype == 'genJulia':
        calculate = calc_genJulia
    elif ftype == 'burning':
        calculate = calc_burning
    elif ftype == 'mandelbar':
        calculate = calc_mandelbar
    else:
        print("invalid fractal name")
        return 0

    for xcol in range(res):
        for ycol in range(res):
            n = calculate(x[xcol] + 1j*y[ycol], maxiter, z0)
            n3[ycol, xcol] = n

    # Set one pixel to 0 so that the colors are standardized
    n3[0,0] = 0
    return x, y, n3


def sierpinski(iteration, shape, scale):
    # Settings
    plt.figure(figsize=(8, 7))
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    # Parameters
    iteration = iteration
    polygon = []
    shape = shape
    scale = scale

    # Create triangle
    for x in range(1, shape+1):
        polygon.append((np.sin(2*np.pi*x/shape), np.cos(2*np.pi*x/shape)))
    plt.plot([x[0] for x in polygon], [y[1] for y in polygon], '.')

    # Initial point
    points = [(np.random.random(), np.random.random())]
    # Generate points
    P = 0
    for i in range(iteration):
        allowedPs = [x for x in range(shape)]
        #if i%3 == 0:
        allowedPs.remove(P)#+2)%shape)
        #else:
        #    allowedPs.remove((P + 3) % shape)
        P = np.random.choice(allowedPs)
        triP = polygon[P]
        prevP = points[-1]
        deltaX = triP[0] - prevP[0]
        deltaY = triP[1] - prevP[1]
        point = (prevP[0] + deltaX*scale, prevP[1] + deltaY*scale)
        points.append(point)

    # plot points
    plt.plot([x[0] for x in points], [y[1] for y in points], '.', c="r", ms=0.2)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.show()


def menger(iteration):
    # Settings
    plt.figure(figsize=(8, 7))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    iteration = iteration
    size = 3**iteration
    menger = np.empty((size, size))

    for i in range(0, iteration+1):
        res = 3**(iteration-i)
        for x in range(1, 3**i, 3):
            for y in range(1, 3**i, 3):
                menger[y*res : (y + 1)*res, x*res : (x + 1)*res] = 100

    plt.imshow(menger,  extent=[-1, 1, -1, 1], cmap="Blues")
    plt.xticks([])
    plt.yticks([])

    plt.show()


def escape_time(z0, ftype):
    if ftype == 'mandelbrot':
        xmin = -2
        xmax = 2
        ymin = -2
        ymax = 2
        figsize = (2.6*2.5, 2.4*2.5)
    elif ftype == 'julia':
        xmin = -1.5 # -1.06
        xmax = 1.5 # -0.9
        ymin = -1.5 #0.26
        ymax = 1.5 #0.38
        figsize = (3*2, 3*2)
    elif ftype == 'genJulia':
        xmin = -1.5  # -1.06
        xmax = 1.5  # -0.9
        ymin = -1.5  # 0.26
        ymax = 1.5  # 0.38
        figsize = (3 * 2, 3 * 2)
    elif ftype == 'burning':
        point = (-1.77, -0.03)
        bound = 1E-2
        xmin = point[0] - bound
        xmax = point[0] + bound
        ymin = point[1] - bound
        ymax = point[1] + bound
        figsize = (3*2, 3*2)
    elif ftype == 'mandelbar':
        xmin = -2  # -1.06
        xmax = 2  # -0.9
        ymin = -2  # 0.26
        ymax = 2  # 0.38
        figsize = (3*2, 3*2)
    elif ftype == 'sierpinski':
        iteration = 80000
        shape = 5
        scale = 0.5
        sierpinski(iteration, shape, scale)
        return 0
    elif ftype == 'menger':
        iteration = 6
        menger(iteration)
        return 0
    else:
        print('invalid fractal type')
        return 0

    res = 500      # resolution per side
    iterations = 50
    cmap = "magma_r"    # choose colormap template. Examples: jet, plasma, viridis, rainbow, hot, inferno, magma, cividis

    startT = time.time()

    print("calculating mandelbrot set...")
    x, y, graph = calc_etfractal(xmin, xmax, ymin, ymax, res, iterations, z0, ftype)
    print("mandelbrot set calculated\nplotting...")
    midT = time.time()

    plt.figure(figsize=figsize)

    plt.imshow(graph, extent=[xmin, xmax, ymin, ymax], cmap=cmap, interpolation='nearest')

    print("graphing done.")
    endT = time.time()

    print("total time =", endT - startT, "s\ncomputing =", midT - startT, "s\ngraphing =", endT - midT, "s")

    if savefig: plt.savefig("{}.png".format(ftype), bbox_inches='tight', dpi=700)
    plt.show()


z0 = -0.01 + 1j
ftype = 'mandelbrot'
savefig = False

escape_time(z0, ftype)

# _____________________________________________________________________________________________________________________
'''
references:
Colormap:       https://matplotlib.org/tutorials/colors/colormaps.html
Julia set:      https://en.wikipedia.org/wiki/Julia_set
Mandelbrot set: https://en.wikipedia.org/wiki/Mandelbrot_set
Burning ship fractal: https://en.wikipedia.org/wiki/Burning_Ship_fractal
Lyapunov fractal:     https://en.wikipedia.org/wiki/Burning_Ship_fractal
Buddhabrot:     https://en.wikipedia.org/wiki/Buddhabrot
Mandelbar:      https://en.wikipedia.org/wiki/Tricorn_(mathematics)
Newton fractal code:  http://code.activestate.com/recipes/577166-newton-fractals/
Newton fractal: https://en.wikipedia.org/wiki/Newton_fractal
https://en.wikipedia.org/wiki/Orbit_trap
'''