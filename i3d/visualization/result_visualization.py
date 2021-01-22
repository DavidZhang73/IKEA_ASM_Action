import json
import os

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm import tqdm


def visualize(result_pathname, frames_path, output_pathname=None):
    fps = 30
    with open(result_pathname, 'r', encoding='utf8') as f:
        result = json.load(f)['results'][frames_path]

    labels = []
    for segment in result:
        label = segment["label"]
        start_frame, end_frame = segment["segment"]
        for _ in range(start_frame, end_frame + 1):
            labels.append(label)

    fig = plt.figure()
    gs = fig.add_gridspec(5, 1)
    ax_video = fig.add_subplot(gs[:-1, :])
    plt.axis("off")
    plt.tight_layout()
    ax_text = fig.add_subplot(gs[-1, :])
    plt.axis("off")
    plt.tight_layout()

    frame_list = [os.path.join(frames_path, frame) for frame in os.listdir(frames_path)]
    frame = cv2.imread(frame_list[0])
    img = ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    txt = ax_text.text(0, 0, "", fontsize=20)

    bar = tqdm(total=len(frame_list))

    def animate(frame_idx):
        frame = cv2.imread(frame_list[frame_idx])
        bar.update(1)
        if frame_idx < len(frame_list):
            img.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            txt.set_text(labels[frame_idx] if frame_idx < len(labels) else "default")
            return img, txt
        else:
            plt.close()
            bar.close()

    anim = animation.FuncAnimation(fig, func=animate, frames=len(frame_list), interval=1000 / fps, blit=False)

    if output_pathname:
        writer = animation.writers["ffmpeg"]
        writer = writer(fps=fps, metadata=dict(artist="DavidZ"), bitrate=-1)
        anim.save(output_pathname, writer=writer)
    else:
        plt.show()


if __name__ == '__main__':
    visualize(
        result_pathname="i3d\\log\\overlap_clips\\demo16_s_fs2\\results\\action_segments.json",
        frames_path="D:\\dataset\\ikea_action_dataset_frame\\yicong",
        output_pathname="i3d\\log\\overlap_clips\\demo16_s_fs2\\results\\visualization.mp4"
    )
