"""Load functions."""
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import streamlit as st
import numpy as np

from spgolfsw.torch.torch_classes import ToTensor, Normalize, EventDetector
from spgolfsw.torch.video_classes import VideoDataset
from spgolfsw.torch.torch_utils import create_images
from spgolfsw.general.utils import get_package_path


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_stuff(uploaded_files, uploaded_filesCOPY):
    return load_stuff_inner(uploaded_files, uploaded_filesCOPY)


def load_stuff_inner(uploaded_files, uploaded_filesCOPY):
    seq_length = 25

    ds = VideoDataset(
        uploaded_files,
        transform=transforms.Compose(
            [ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        ),
    )

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False,
    )
    try:
        PACKAGE_PATH = get_package_path()
        save_dict = torch.load(
            PACKAGE_PATH.joinpath("models/swingnet_1800.pth.tar"),
            map_location=torch.device("cpu"),
        )
    except ValueError as e:
        print(
            "Model weights not found. "
            "Download model weights and place in 'models'"
            f" folder. See README for instructions {e}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.load_state_dict(save_dict["model_state_dict"])
    model.to(device)
    model.eval()
    print("Loaded model weights")

    print("Testing...")
    for sample in dl:
        images = sample["images"]
        # full samples do not fit into GPU memory
        # so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length :, :, :, :]
            else:
                image_batch = images[
                    :, batch * seq_length : (batch + 1) * seq_length, :, :, :
                ]
            logits = model(image_batch)
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    print("Predicted event frames: {}".format(events))

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print("Confidence: {}".format([np.round(c, 3) for c in confidence]))

    imgALL, fimg = create_images(uploaded_filesCOPY, events)

    return events, imgALL, fimg
