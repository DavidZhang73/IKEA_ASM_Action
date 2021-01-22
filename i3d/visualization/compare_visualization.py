import json

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm import tqdm

from vidat import Vidat


def visualize(annotation_pathname, video_pathname, result_pathname, frames_path, output_pathname=None):
    vidat = Vidat(annotation_pathname, video_pathname)

    action_list = [action["name"] for action in vidat.config.action_label]

    video = vidat.annotation.video
    fps = video.fps
    frames = video.frames

    actions = vidat.annotation.actions

    gt_labels = [[0] for _ in range(frames)]
    for action in actions:
        no = action_list.index(action.action["name"])
        start = int(action.start * fps)
        end = int(action.end * fps)
        for i in range(start, end):
            a = gt_labels[i]
            if a[0] == 0:
                gt_labels[i][0] = no
            else:
                gt_labels[i].append(no)

    with open(result_pathname, 'r', encoding='utf8') as f:
        result = json.load(f)['results'][frames_path]

    labels = []
    for segment in result:
        label = segment["label"]
        start_frame, end_frame = segment["segment"]
        for _ in range(start_frame, end_frame + 1):
            labels.append(label)

    fig = plt.figure()
    gs = fig.add_gridspec(6, 1)
    ax_video = fig.add_subplot(gs[:-2, :])
    plt.axis("off")
    plt.tight_layout()
    ax_text = fig.add_subplot(gs[-2, :])
    plt.axis("off")
    plt.tight_layout()
    ax_text_gt = fig.add_subplot(gs[-1, :])
    plt.axis("off")
    plt.tight_layout()

    cap = cv2.VideoCapture(video_pathname)
    flag, frame = cap.read()
    img = ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    txt = ax_text.text(0, 0, "pred:", fontsize=20)
    txt_gt = ax_text_gt.text(0, 0, "gt    :", fontsize=20)

    bar = tqdm(total=frames)

    def animate(frame_idx):
        flag, frame = cap.read()
        bar.update(1)
        if flag:
            img.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            txt.set_text("pred: " + labels[frame_idx] if frame_idx < len(labels) else "default")
            txt_gt.set_text("gt    : " + ", ".join([action_list[i] for i in gt_labels[frame_idx]]))
            return img, txt, txt_gt
        else:
            plt.close()
            bar.close()

    anim = animation.FuncAnimation(fig, func=animate, frames=frames, interval=1000 / fps, blit=False)

    if output_pathname:
        writer = animation.writers["ffmpeg"]
        writer = writer(fps=fps, metadata=dict(artist="DavidZ"), bitrate=-1)
        anim.save(output_pathname, writer=writer)
    else:
        plt.show()


if __name__ == '__main__':
    visualize(
        annotation_pathname="D:\\dataset\\ikea_action_dataset_video\\yicong\\annotations.json",
        video_pathname="D:\\dataset\\ikea_action_dataset_video\\yicong\\scan_video.mp4",
        result_pathname="i3d\\log\\overlap_clips\\demo16_s_fs2\\results\\action_segments.json",
        frames_path="D:\\dataset\\ikea_action_dataset_frame_small\\yicong",
        output_pathname="i3d\\log\\overlap_clips\\demo16_s_fs2\\results\\compare_visualization.mp4"
    )
