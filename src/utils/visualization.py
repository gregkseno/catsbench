import numpy as np
from matplotlib.figure import Figure
from PIL import Image


def fig2img(fig: Figure) -> Image.Image:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis = 2)
    return Image.frombytes("RGBA", (w, h), buf.tobytes())
