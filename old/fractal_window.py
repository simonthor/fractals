import tkinter as tk
import fractals


def mandelbrot():
    fractal = sierpinski.fractal(3, 0.5, 10000)
    fractal.plot()


buttonCursor = "hand2"
h = w = 500
bg = "white"

top = tk.Tk()
top.configure(bg=bg)

top.title("Fractals")
top.geometry("%dx%d" % (w, h))

frame = tk.Frame(top, bg=bg)
frame.pack(expand=True, fill=tk.BOTH)

tk.Label(frame, text="Choose a fractal to generate: ", bg=bg).place(x=140, y=100)
mandelbrotB = tk.Button(frame, text="mystisk knapp", command=mandelbrot)
mandelbrotB.place(x=180, y=200)

top.mainloop()
