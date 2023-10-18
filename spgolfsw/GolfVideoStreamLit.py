"""Streamlit app to analyse golf swings."""
import scipy.io

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import copy
import sys

from spgolfsw.torch.loads import load_stuff


def main():
    st.title("Golf Swing")
    # loada = st.checkbox('Load',key='AC')
    XxX = sorted(
        [(x, sys.getsizeof(globals().get(x))) for x in dir()],
        key=lambda x: x[1],
        reverse=True,
    )
    memos = np.array([(sys.getsizeof(globals().get(x))) for x in dir()])

    print("Main", XxX, "---", np.shape(XxX))
    st.write("Memory", np.sum(memos))

    # if loada:
    uploaded_files = st.sidebar.file_uploader(
        "Choose video", accept_multiple_files=False
    )

    uploaded_filesCOPY = copy.copy(uploaded_files)

    if uploaded_files:
        events, imgALL, fimg = load_stuff(uploaded_files, uploaded_filesCOPY)
    del uploaded_files

    plota = st.checkbox("Plot", key="AC2")

    if plota:
        print(
            "PlotBox",
            sorted(
                [(x, sys.getsizeof(globals().get(x))) for x in dir()],
                key=lambda x: x[1],
                reverse=True,
            ),
        )

        imgSEL = st.selectbox("Select Image", fimg)

        numSEL = [oo for oo, x in enumerate(fimg) if x == imgSEL][0]

        f = plt.figure(figsize=(6, 6))

        plt.imshow(imgALL[numSEL])

        st.pyplot(f)


if __name__ == "__main__":
    main()
