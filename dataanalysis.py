import numpy as np
import matplotlib.pyplot as plt


class DataExploration:
    def __init__(self) -> None:
        pass

    def plot_density(self, values, bin_width=0.05, xlabel="Values"):
        # Calculate histogram values
        bins = np.arange(min(values), max(values) + bin_width, bin_width)

        plt.hist(values, bins=bins, density=True, alpha=0.7)

        plt.xlabel(f"{xlabel}")
        plt.ylabel("Density")
        plt.title(f"Density of {xlabel}")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plt.show()
