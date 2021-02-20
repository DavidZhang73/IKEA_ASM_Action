import json

import numpy as np
from sklearn.metrics import classification_report, average_precision_score

from vidat import Vidat


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


def evaluate(
        annotation_pathname,
        result_pathname,
        frames_path
):
    # Get labels
    action_list, gt_labels, predict_labels = get_label(annotation_pathname, result_pathname, frames_path)

    print("Action List:", [action_list[clazz] for clazz in np.unique(gt_labels)], sep="\n")

    # for clazz in class_list:
    #     tp = np.sum(np.logical_and(predict_labels == clazz, gt_labels == clazz))
    #     tn = np.sum(np.logical_and(predict_labels != clazz, gt_labels != clazz))
    #     fp = np.sum(np.logical_and(predict_labels == clazz, gt_labels != clazz))
    #     fn = np.sum(np.logical_and(predict_labels != clazz, gt_labels == clazz))
    #     total = tp + tn + fp + fn
    #     recall = tp / (tp + fn)
    #     precision = tp / (tp + fp)
    #     accuracy = tp / total
    #     print("-" * 50)
    #     print(f"class: {action_list[clazz]}\nAccuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}")
    #     print("-" * 50)

    # Evaluation

    print("Classification Report:", classification_report(gt_labels, predict_labels, zero_division=1), sep="\n")

    ap_list = []
    class_list = np.unique(gt_labels)
    for clazz in class_list:
        y_true = gt_labels == clazz
        y_score = predict_labels == clazz
        ap_list.append(average_precision_score(y_true, y_score))
    print("AP:", ap_list, sep="\n")
    print("mAP:", np.mean(ap_list), sep="\n")


if __name__ == '__main__':
    evaluate(
        annotation_pathname="D:\\dataset\\ikea_action_dataset_video\\yicong\\annotations.json",
        result_pathname="i3d\\log\\overlap_clips7\\demo16_s_fs2\\results\\action_segments.json",
        frames_path="D:\\dataset\\ikea_action_dataset_frame_small\\yicong",
    )
