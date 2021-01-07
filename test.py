from torchvision import transforms

from IKEAActionDataset import IKEAActionVideoClipDataset as Dataset
from i3d import videotransforms

train_transform = transforms.Compose(
    [
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip()
    ]
)

dataset = Dataset(
    dataset_path=r'D:\dataset\ikea_action_dataset_frame',
    annotation_path=r'D:\dataset\ikea_action_dataset_video',
    transform=train_transform,
    index_filename="test_dataset_index.txt",
    frames_per_clip=16,
    frame_skip=2,
)

# for item in dataset[0]:
#     print(item)
print(len(dataset))
