"""One plotting style for every figure in the report."""

import matplotlib.pyplot as plt

PALETTE = ["#3B6FB6", "#E07B39", "#4E9F6E", "#B5487F"]  # glioma, meningioma, no tumour, pituitary


def apply():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "legend.frameon": False,
        }
    )
