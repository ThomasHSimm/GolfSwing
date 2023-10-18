"""Utility torch functions."""
import torch.nn as nn
import cv2
import tempfile


def create_images(file_path: str, events):
    """
    Get images from a video at set positions.

    Given a video file location (fila) it will save as images to a folder
    Given positions in video (pos) these images from the video are saved
    pos is created based on positions of swings.

    Parameters:
        file_path (str):
    Returns:

    """

    tfile2 = tempfile.NamedTemporaryFile(delete=False)
    tfile2.write(file_path.read())
    cap = cv2.VideoCapture(tfile2.name)

    imgALL = []
    fimg = []
    for e in events:
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        imgALL.append(img)
        fimg.append(e)

    cap.release()
    return imgALL, fimg


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )
