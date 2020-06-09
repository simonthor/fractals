from render import animate
import numpy as np

exp = np.linspace(0, 10, 100)
animate('test_video.mp4', len(exp), 'jet', fps=10,
        iter_args={'exp': exp},
        const_args={'iterations': 50, 're_lim': (-2, 2), 'im_lim': (-2, 2), 'resolution': 400, 'et_function': lambda z, c, exp: z**exp + c})
