import sys
import os
import cv2
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from computeFlow import compute_flow
from trackingTester import tracking_tester
from signAcademicPolicy import sign_academic_honesty_policy


def runHw7(*args):
    """
    runHw7 is the "main" interface that lets you execute all the
    challenges in homework 7. It lists a set of functions corresponding
    to the problems that need to be solved.
    """

    fun_handles = [
        honesty,
        debug1a, challenge1a,
        challenge2a, challenge2b, challenge2c
    ]

    runTests(args, fun_handles)

# --------------------------------------------------------------------------
# Academic Honesty Policy
# --------------------------------------------------------------------------


def honesty():
    # Type your full name and uni (both in string) to state your agreement
    # to the Code of Academic Integrity.
    sign_academic_honesty_policy('Amogh Sudhir Dixit', 'asdixit3')

# --------------------------------------------------------------------------
# Tests for Challenge 1: Optical flow using template matching
# --------------------------------------------------------------------------


def debug1a():
    img1 = io.imread('simple1.png')
    img2 = io.imread('simple2.png')

    search_radius = 20      # Half size of the search window
    template_radius = 10     # Half size of the template window
    grid_MN = [30, 30]      # Number of rows and cols in the grid

    result = compute_flow(img1, img2, search_radius, template_radius, grid_MN)
    io.imsave('simpleresult.png', result)


def challenge1a():
    img_list = ['flow1.png', 'flow2.png', 'flow3.png',
                'flow4.png', 'flow5.png', 'flow6.png']
    img_stack = [io.imread(fname) for fname in img_list]

    search_radius = 40     # Half size of the search window
    template_radius = 30   # Half size of the template window
    grid_MN = [24, 32]   # Number of rows and cols in the grid

    for i in range(1, len(img_stack)):
        result = compute_flow(img_stack[i-1], img_stack[i],
                              search_radius, template_radius, grid_MN)
        io.imsave(f'result{i}_{i+1}.png', result)

# --------------------------------------------------------------------------
# Tests for Challenge 2: Tracking with color histogram template
# --------------------------------------------------------------------------


def challenge2a():
    data_params = {
        'data_dir': 'walking_person',
        'out_dir': 'walking_person_result',
        'video_out_dir': 'video_results',
        'video_out_fname': 'walking_person.mp4',
        'frame_ids': list(range(1, 251)),
        'genFname': lambda x: f'frame{x}.png'
    }

    tracking_params = {}

    # only temporary to help choose the target! Comment out before submission!
    # tracking_params['rect'] = select_rectangle(data_params)

    # TODO: Replace select_rectangle with actual parameters in submission
    tracking_params['rect'] = [187, 58, 49, 133]

    # Half size of the search window
    tracking_params['search_radius'] = 15  # TODO
    # Number of bins in the color histogram
    tracking_params['bin_n'] = 32  # TODO

    tracking_tester(data_params, tracking_params)
    generate_video(data_params)


def challenge2b():
    data_params = {
        'data_dir': 'rolling_ball',
        'out_dir': 'rolling_ball_result',
        'video_out_dir': 'video_results',
        'video_out_fname': 'rolling_ball.mp4',
        'frame_ids': list(range(1, 251)),
        'genFname': lambda x: f'frame{x}.png'
    }

    tracking_params = {}
    # only temporary to help choose the target! Comment out before submission!
    # tracking_params['rect'] = select_rectangle(data_params)

    # TODO: Replace select_rectangle with actual parameters in submission
    tracking_params['rect'] = [148, 121, 57, 63]

    tracking_params['search_radius'] = 15  # TODO
    tracking_params['bin_n'] = 32  # TODO

    tracking_tester(data_params, tracking_params)
    generate_video(data_params)


def challenge2c():
    data_params = {
        'data_dir': 'basketball',
        'out_dir': 'basketball_result',
        'video_out_dir': 'video_results',
        'video_out_fname': 'basketball.mp4',
        'frame_ids': list(range(1, 251)),
        'genFname': lambda x: f'frame{x}.png'
    }

    tracking_params = {}
    # only temporary to help choose the target! Comment out before submission!
    # tracking_params['rect'] = select_rectangle(data_params)

    # TODO: Replace select_rectangle with actual parameters in submission
    tracking_params['rect'] = [310, 221, 39, 115]

    tracking_params['search_radius'] = 20  # TODO
    tracking_params['bin_n'] = 32  # TODO

    tracking_tester(data_params, tracking_params)
    generate_video(data_params)

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------


def select_rectangle(data_params):
    """
    Display an image and let the user draw a rectangle by click-drag-release.
    Returns ((x0, y0), (x1, y1)) pixel coordinates.
    """

    img_path = os.path.join(data_params['data_dir'],
                            data_params['genFname'](data_params['frame_ids'][0]))
    image = io.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Drag to select a rectangle")
    plt.axis('on')

    rect = Rectangle((0, 0), 0, 0, linewidth=1.5,
                     edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    start = [None, None]
    end = [None, None]
    pressed = [False]

    def on_press(event):
        if event.inaxes != ax:
            return
        pressed[0] = True
        start[0], start[1] = event.xdata, event.ydata
        rect.set_xy((start[0], start[1]))
        rect.set_width(0)
        rect.set_height(0)
        fig.canvas.draw_idle()

    def on_motion(event):
        if not pressed[0] or event.inaxes != ax:
            return
        x0, y0 = start
        x1, y1 = event.xdata, event.ydata
        if x1 is None or y1 is None:
            return
        rect.set_xy((min(x0, x1), min(y0, y1)))
        rect.set_width(abs(x1 - x0))
        rect.set_height(abs(y1 - y0))
        fig.canvas.draw_idle()

    def on_release(event):
        if not pressed[0] or event.inaxes != ax:
            return
        pressed[0] = False
        end[0], end[1] = event.xdata, event.ydata
        plt.close(fig)

    cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
    cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
    cid_release = fig.canvas.mpl_connect('button_release_event', on_release)

    plt.show()

    if None in start or None in end:
        raise RuntimeError("No rectangle selected.")

    x0, y0 = start
    x1, y1 = end
    xmin, ymin = int(min(x0, x1)), int(min(y0, y1))
    width, height = int(abs(x1 - x0)), int(abs(y1 - y0))

    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    rect_out = [xmin, ymin, width, height]
    print('===========')
    print(f'[xmin ymin width height] = {rect_out}')
    print('===========')
    return rect_out
    return (start[0], start[1]), (end[0], end[1])


def generate_video(data_params):
    os.makedirs(data_params['video_out_dir'], exist_ok=True)

    n_frames = len(data_params['frame_ids'])
    video_filename = os.path.join(
        data_params['video_out_dir'], data_params['video_out_fname'])

    frame0 = cv2.imread(os.path.join(data_params['out_dir'],
                                     data_params['genFname'](data_params['frame_ids'][0])))

    print(
        f'Writing {len(data_params["frame_ids"])} frames to video file {video_filename}...')

    if frame0 is None:
        print("No frames found to generate video.")
        return
    height, width, _ = frame0.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))

    for fid in data_params['frame_ids']:
        fname = os.path.join(
            data_params['out_dir'], data_params['genFname'](fid))
        frame = cv2.imread(fname)
        if frame is None:
            print("Could not read:", fname)
            continue

        # Ensure same dimensions
        if frame.shape[0] != height or frame.shape[1] != width:
            print("Resizing:", fname, "from",
                  frame.shape, "to", (height, width))
            frame = cv2.resize(frame, (width, height))

        # Ensure 3 channels
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        video.write(frame)

    video.release()


def runTests(args, fun_handles):
    if not args:
        print("Registered functions:")
        for f in fun_handles:
            print(f.__name__)
        return
    if args[0] == 'all':
        for f in fun_handles:
            print(f'Running {f.__name__}...')
            f()
    else:
        name = args[0]
        for f in fun_handles:
            if f.__name__ == name:
                print(f'Running {name}...')
                f()
                return
        print(f'Function {name} not found.')


if __name__ == "__main__":
    runHw7(*sys.argv[1:])
