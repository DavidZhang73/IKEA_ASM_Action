import json
import subprocess
import time
from itertools import combinations
from os import path, chdir, getcwd

import numpy as np
from sklearn.metrics import classification_report, average_precision_score
from tqdm import tqdm

from vidat import Vidat

CWD = getcwd()
LOG_PATH = path.join(CWD, "log", f"{int(time.time() * 1000)}.log")

TRAIN_INDEX_PATHNAME = r"D:\dataset\ikea_action_dataset_frame_small\train_dataset_index.txt"
TEST_INDEX_PATHNAME = r"D:\dataset\ikea_action_dataset_frame_small\test_dataset_index.txt"


def run(cmd, cwd=None, log_prefix=None):
    if cwd:
        chdir(cwd)
    try:
        out_bytes = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        out_bytes = e.output
    out_text = out_bytes.decode('utf-8')
    with open(LOG_PATH, 'a', encoding="utf-8") as f:
        f.write('\n\n' + log_prefix + '\n\n' + out_text)
    if cwd:
        chdir(CWD)


def get_label(annotation_pathname, result_pathname, frames_path):
    # Get labels
    vidat = Vidat(annotation_pathname)

    action_list = [action["name"] for action in vidat.config.action_label]

    video = vidat.annotation.video
    fps = video.fps
    frames = video.frames
    actions = vidat.annotation.actions

    gt_labels = np.zeros(frames, dtype=int)
    for action in actions:
        no = action_list.index(action.action["name"])
        start = int(action.start * fps)
        end = int(action.end * fps)
        for i in range(start, end):
            gt_labels[i] = no

    with open(result_pathname, 'r', encoding='utf8') as f:
        result = json.load(f)['results'][frames_path]

    predict_labels = np.zeros(frames, dtype=int)
    for segment in result:
        label = segment["label"]
        start_frame, end_frame = segment["segment"]
        for i in range(start_frame, end_frame + 1):
            predict_labels[i] = (action_list.index(label))

    # Remove reverse
    gt_labels[gt_labels == 7] = 2
    predict_labels[predict_labels == 7] = 2
    gt_labels[gt_labels == 8] = 3
    predict_labels[predict_labels == 8] = 3

    action_list.remove(action_list[7])
    action_list.remove(action_list[7])

    gt_labels[gt_labels == 9] = 7
    predict_labels[predict_labels == 9] = 7

    return action_list, gt_labels, predict_labels


# prepare dataset
video_list = [
    "haodong",
    "jack",
    "liyuan",
    "reaching",
    "yicong",
    "zheyu",
]
combination_list = list(combinations(video_list, 2))

bar = tqdm(combination_list)
for test_index_list in bar:
    train_index_list = [video for video in video_list if video not in test_index_list]
    bar.set_description(str(test_index_list) + str(train_index_list))
    run_name = '_'.join(test_index_list)

    # with open(TRAIN_INDEX_PATHNAME, 'w') as f:
    #     f.write("\n".join(train_index_list))
    # with open(TEST_INDEX_PATHNAME, 'w') as f:
    #     f.write("\n".join(test_index_list))
    #
    # # train the model
    #
    # run(cmd=[
    #     "python",
    #     "train_i3d.py"
    # ],
    #     cwd=r"E:\Projects\Other\jupyterlab\notebook\COMP 4550\projects\IKEA_ASM_Action\i3d",
    #     log_prefix=run_name
    # )
    #
    # # test the model
    #
    # run(cmd=[
    #     "python",
    #     "test_i3d.py"
    # ],
    #     cwd=r"E:\Projects\Other\jupyterlab\notebook\COMP 4550\projects\IKEA_ASM_Action\i3d",
    #     log_prefix=run_name
    # )
    #
    # # prepare for next run
    #
    # rename(
    #     r"E:\Projects\Other\jupyterlab\notebook\COMP 4550\projects\IKEA_ASM_Action\i3d\log\overlap_clips\demo16_s_fs2",
    #     path.join(
    #         r"E:\Projects\Other\jupyterlab\notebook\COMP 4550\projects\IKEA_ASM_Action\i3d\log\overlap_clips",
    #         run_name
    #     )
    # )

    # evaluation
    gt_labels = []
    predict_labels = []
    for test_video in test_index_list:
        action_list, _gt_labels, _predict_labels = get_label(
            annotation_pathname=f"D:\\dataset\\ikea_action_dataset_video\\{test_video}\\annotations.json",
            result_pathname=
            f"E:\\Projects\\Other\\jupyterlab\\notebook\\COMP 4550\\projects\\IKEA_ASM_Action\\i3d\\log\\overlap_clips\\{run_name}\\results\\action_segments.json",
            frames_path=f"D:\\dataset\\ikea_action_dataset_frame_small\\{test_video}",
        )
        gt_labels.extend(_gt_labels)
        predict_labels.extend(_predict_labels)

    _classification_report = classification_report(gt_labels, predict_labels, zero_division=1, output_dict=True)

    ap_list = []
    class_list = np.unique(gt_labels)
    for clazz in class_list:
        y_true = gt_labels == clazz
        y_score = predict_labels == clazz
        ap_list.append(average_precision_score(y_true, y_score))
    with open(
            f"E:\\Projects\\Other\\jupyterlab\\notebook\\COMP 4550\\projects\\IKEA_ASM_Action\\i3d\\log\\overlap_clips\\{run_name}\\results\\evaluation.json",
            'w', encoding='utf-8') as f:
        json.dump({
            "action_list": action_list,
            "classification_report": _classification_report,
            "ap_list": ap_list
        }, f)
