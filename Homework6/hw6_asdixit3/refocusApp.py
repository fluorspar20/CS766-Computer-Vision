import numpy as np
import matplotlib.pyplot as plt


def refocus_app(rgb_stack, depth_map):
    H, W = depth_map.shape
    N = rgb_stack.shape[-1] // 3

    def get_image(idx):
        """Return the RGB image at focal-stack index idx."""
        idx = int(np.clip(idx, 0, N - 1))
        return rgb_stack[:, :, 3 * idx:3 * (idx + 1)]

    current_idx = 0

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(get_image(current_idx))
    ax.set_title(f"Click a point to refocus (layer {current_idx}). "
                 f"Press 'enter' to quit.")
    plt.draw()

    while True:
        pts = plt.ginput(1, timeout=-1)

        # user pressed 'enter'
        if len(pts) == 0:
            break

        x, y = pts[0]
        xi, yi = int(round(x)), int(round(y))

        target_idx = int(depth_map[yi, xi])
        target_idx = max(0, min(N - 1, target_idx))

        # transition
        if target_idx != current_idx:
            step = 1 if target_idx > current_idx else -1
            for i in range(current_idx + step, target_idx + step, step):
                ax.imshow(get_image(i))
                ax.set_title(f"Transitioning... layer {i}")
                ax.axis("off")
                plt.pause(0.05)

        current_idx = target_idx
        ax.imshow(get_image(current_idx))
        ax.set_title(f"Focused at ({xi}, {yi}) -> layer {current_idx}. "
                     f"Press 'enter' to quit.")
        ax.axis("off")
        plt.draw()

    plt.close(fig)
