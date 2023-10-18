"""Classes to handle videos functionality."""
from torch.utils.data import Dataset

import cv2
import numpy as np
import tempfile


class VideoDataset(Dataset):
    """
    Handle a video as images for use with Torch DataLoader.
    """

    def __init__(self, uploaded_files, input_size=160, transform=None):
        self.input_size = input_size
        self.transform = transform
        self.uploaded_files = uploaded_files

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # If uploaded_files is the file_path
        if type(self.uploaded_files) == str:
            cap = cv2.VideoCapture(self.uploaded_files)
        # If uploaded_files is a streamlit video object
        else:
            # create a fake file needed for cv2
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(self.uploaded_files.read())
            cap = cv2.VideoCapture(tfile.name)

        frame_size = [
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        ]
        try:
            ratio = self.input_size / max(frame_size)
        except:
            ratio = 1
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # preprocess and return frames
        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=[0.406 * 255, 0.456 * 255, 0.485 * 255],
            )  # ImageNet means (BGR)

            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()

        labels = np.zeros(len(images))  # only for compat with transforms
        sample = {"images": np.asarray(images), "labels": np.asarray(labels)}

        if self.transform:
            sample = self.transform(sample)
        return sample
