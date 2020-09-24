# fractals
The code here is used to generate fractals.

## Functionality
[render.py](render.py) contains 2 functions for simple animation of escape time fractals. 
For more complex fractals animations, use `generate_escape_time` in [fractal_generator.py](fractal_generator.py). 

## Packages
See [environment.yml](environment.yml).

If you use conda (which I recommend but is not required), go to your project directory and run
```bash
conda env create -f environment.yml --prefix ./env
conda activate ./env
```
Then all packages (and correct python version) will install automatically. 

## TODO
- [ ] Add docstrings
- [ ] GPU-accelerated computation (`cupy`?)

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
