



def makeVideo(X_2d,labels):

    gif_name = "test.gif"
    gif_duration = X_2d.shape[0]
    gif_fps = 10

    timed_activations = dataset['timed_activations']
    duration = X_2d.shape[0]

    # gets the figure (here 1) and the axis array (ax1, ax2)
    fig, ax = plt.subplots(1, figsize=(4, 4), facecolor='white')

    def make_frame(t):
        ax.clear()
        print(t)

        ax.set_title("Activations", fontsize=16)
        ax.scatter(X_2d[:,0],X_2d[:,1],alpha=0.1,lw=0.0)

        ax.scatter(X_2d[t*fps:t*fps+1,0],X_2d[t*fps:t*fps+1,1],alpha=1,lw=0.0)

        return mplfig_to_npimage(fig)

    animation = mpy.VideoClip(make_frame, duration = gif_duration)
    animation.write_gif2(gif_name, fps=gif_fps)