"""
Adapter for IKEAActionDataset with vidat
"""
import os

import cv2
import numpy as np
import torch

from vidat import Vidat


class IKEAActionVideoClipDataset:
    def __init__(
            self,
            dataset_path,
            annotation_path,
            transform,
            index_filename,
            annotation_filename='annotations.json',
            frames_per_clip=16,
            frame_skip=2,
    ):
        self.dataset_path = dataset_path
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.frame_skip = frame_skip

        with open(os.path.join(dataset_path, index_filename), 'r') as f:
            self.index_list = [line.strip() for line in f.readlines()]

        # scan the dataset path
        self.action_name_list = []
        self.video_pathname_list = []
        self.video_name_to_annotation = {}
        for video_name in self.index_list:
            video_pathname = os.path.join(dataset_path, video_name)
            if os.path.isdir(video_pathname):
                self.video_pathname_list.append(video_pathname)
                annotation = Vidat(
                    os.path.join(annotation_path, video_name, annotation_filename)
                )
                for label in annotation.config.action_label:
                    # TODO: remove reverse
                    if label['name'].replace(" - rev", "") not in self.action_name_list:
                        # self.action_name_list.append(label['name'])
                        self.action_name_list.append(label['name'])
                self.video_name_to_annotation[video_pathname] = annotation

        # video list
        self.video_list = self.get_video_list()

        # clip list
        self.clip_list, self.clip_label_count = self.get_clip_list()

    def get_video_list(self):
        """
        # Extract the label video from the annotation files
        :return: (video_pathname, multi-label per-frame, number of frames in the video)
        """
        video_list = []

        for video_pathname in self.video_pathname_list:
            annotation = self.video_name_to_annotation[video_pathname]
            num_frames = annotation.annotation.video.frames

            labels = np.zeros((len(self.action_name_list), num_frames), np.float32)  # allow multi-class representation
            labels[0, :] = np.ones((1, num_frames), np.float32)  # initialize all frames as background|transition

            for action in annotation.annotation.actions:
                labels[0, action.start_frame:action.end_frame] = 0  # remove the background label
                # labels[self.action_name_list.index(action.action['name']), action.start_frame:action.end_frame] = 1
                labels[self.action_name_list.index(action.action['name'].replace(" - rev", "")),
                action.start_frame:action.end_frame] = 1
            video_list.append((video_pathname, labels, num_frames))

        return video_list

    def get_clip_list(self):
        clip_dataset = []
        label_count = np.zeros(len(self.action_name_list))
        for i, video in enumerate(self.video_list):
            num_frames = video[2]
            n_clips = int(num_frames / (self.frames_per_clip * self.frame_skip))
            remaining_frames = num_frames % (self.frames_per_clip * self.frame_skip)
            for j in range(0, n_clips):
                for k in range(0, self.frame_skip):
                    start = j * self.frames_per_clip * self.frame_skip + k
                    end = (j + 1) * self.frames_per_clip * self.frame_skip
                    label = video[1][:, start:end:self.frame_skip]
                    label_count = label_count + np.sum(label, axis=1)
                    frame_ind = np.arange(start, end, self.frame_skip).tolist()
                    clip_dataset.append((video[0], label, frame_ind, self.frames_per_clip, i, 0))
            # if not remaining_frames == 0:
            #     frame_pad = self.frames_per_clip - remaining_frames
            #     start = n_clips * self.frames_per_clip * self.frame_skip + self.frame_skip
            #     end = start + remaining_frames
            #     label = video[1][:, start:end:self.frame_skip]
            #     label_count = label_count + np.sum(label, axis=1)
            #     label = video[1][:, start - frame_pad:end:self.frame_skip]
            #     frame_ind = np.arange(start - frame_pad, end, self.frame_skip).tolist()
            #     clip_dataset.append((video[0], label, frame_ind, self.frames_per_clip, i, frame_pad))
        return clip_dataset, label_count

    def load_rgb_frames(self, video_full_path, frame_ind):
        """
        load video file and extract the frames
        :param video_full_path:
        :param frame_ind: index list of frames
        :return: frames
        """
        frames = []
        for i in frame_ind:
            img_filename = os.path.join(video_full_path, str(i).zfill(5) + '.jpg')

            img = cv2.imread(img_filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # TODO: resize
            # if self.resize is not None:
            #     w, h, c = img.shape
            #     if w < self.resize or h < self.resize:
            #         d = self.resize - min(w, h)
            #         sc = 1 + d / min(w, h)
            #         img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            #     img = cv2.resize(img, dsize=(self.resize, self.resize))  # resizing the images
            #     img = (img / 255.) * 2 - 1

            frames.append(img)

        return np.asarray(frames, dtype=np.float32)

    def video_to_tensor(self, pic):
        """
        Convert a numpy.ndarray to tensor.
        Converts a numpy.ndarray (T x H x W x C) to a torch.FloatTensor of shape (C x T x H x W)
        :param pic: (numpy.ndarray) Video to be converted to tensor
        :return: Converted video Tensor
        """
        return torch.tensor(pic.transpose([3, 0, 1, 2]), dtype=torch.float32)

    def __getitem__(self, item):
        video_full_path, labels, frame_ind, n_frames_per_clip, vid_idx, frame_pad = self.clip_list[item]

        img_list = self.load_rgb_frames(video_full_path, frame_ind)
        img_list = self.transform(img_list)

        return self.video_to_tensor(img_list), torch.from_numpy(labels), vid_idx, frame_pad

    def __len__(self):
        return len(self.clip_list)
