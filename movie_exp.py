import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

fig = plt.figure(facecolor="white") # <- ADDED FACECOLOR FOR WHITE BACKGROUND
ax = plt.axes()
x = np.random.randn(10, 1)
y = np.random.randn(10, 1)
p = plt.plot(x, y, 'ko')
time = np.arange(2341973, 2342373)

last_i = None
last_frame = None

def animate(t):
    global last_i, last_frame

    i = int(t)
    if i == last_i:
        return last_frame

    xn = x + np.sin(2 * np.pi * time[i] / 10.0)
    yn = y + np.cos(2 * np.pi * time[i] / 8.0)
    p[0].set_data(xn, yn)

    last_i = i
    last_frame = mplfig_to_npimage(fig)
    return last_frame

duration = len(time)
fps = 15
animation = mpy.VideoClip(animate, duration=duration)
animation.write_gif2("test.mp4", fps=fps)