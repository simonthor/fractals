# fractals
The code here is used to generate fractals.

The project is currently being rewritten from scratch for easier maintainability and better features. 

## Functionality
The functionality below only describes the old code in [old](./old/).
- interactive_fractal.py file can have interactive escape-time fractals, meaning that the user can zoom in and out by clicking on the image.
- fractals.py provides more extensive options of fractals, including (but not limited to) sierpinski's triangle and the Koch snowflake (sort of).

## Prerequisites
See [environment.yml](environment.yml).

## TODO
- [ ] SymPy support for `et_function`.
- [ ] GPU-accelerated computation (cupy?)
- [ ] AI for image identification
- [ ] Test unique value variance calculation instead of variance calculation (although `unique` method is likely much slower).

## References
- Colormap:             https://matplotlib.org/tutorials/colors/colormaps.html
- Julia set:            https://en.wikipedia.org/wiki/Julia_set
- Mandelbrot set:       https://en.wikipedia.org/wiki/Mandelbrot_set
- Burning ship fractal: https://en.wikipedia.org/wiki/Burning_Ship_fractal
- Lyapunov fractal:     https://en.wikipedia.org/wiki/Burning_Ship_fractal
- Buddhabrot:           https://en.wikipedia.org/wiki/Buddhabrot
- Mandelbar:            https://en.wikipedia.org/wiki/Tricorn_(mathematics)
- Newton fractal code:  http://code.activestate.com/recipes/577166-newton-fractals/
- Newton fractal:       https://en.wikipedia.org/wiki/Newton_fractal
