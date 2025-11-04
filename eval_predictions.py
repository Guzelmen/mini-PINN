import numpy as np
import matplotlib.pyplot as plt
import pickle
import math


def plot_predictions(filename):
    """
    Plots the predictions of the model, psi(x) vs x.
    Fits a polynomial to the predictions and plots it.
    Makes a big figure with all plots for specific epochs.

    Inputs: pkl file with structure:
        {
            "epoch_a": {"inputs": np.ndarray(...), "outputs": np.ndarray(...)},
            "epoch_b": {"inputs": np.ndarray(...), "outputs": np.ndarray(...)},
            ...
        }

    Returns: big figure with all plots for specific epochs, saved in plots/filename.png.
    """

    pkl = f"predictions/{filename}.pkl"
    with open(pkl, "rb") as f:
        data = pickle.load(f)

    epochs = list(data.keys())

    n_epochs = len(epochs)
    cols = math.ceil(math.sqrt(n_epochs * 1.5))
    rows = math.ceil(n_epochs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for ax, e in zip(axes, epochs):

        inputs = data[e]["inputs"].flatten()
        outputs = data[e]["outputs"].flatten()

        # Sort inputs and reorder outputs to preserve correspondence
        sort_idx = np.argsort(inputs)
        inputs_sorted = inputs[sort_idx]
        outputs_sorted = outputs[sort_idx]

        ax.plot(inputs_sorted, outputs_sorted, '+',
                color="black", label='Model Predictions')

        p, cov = np.polyfit(inputs_sorted, outputs_sorted, 3, cov=True)
        err = np.sqrt(np.diag(cov))
        # Use a simple label for the plot, then modify the legend text after
        ax.plot(inputs_sorted, np.polyval(p, inputs_sorted),
                color="red", label='Fit')

        ax.set_xlabel('x')
        ax.set_ylabel(r'$\psi(x)$')
        ax.set_title(f"Epoch: {e}")
        # Place legend inside the plot
        legend = ax.legend(loc='best', framealpha=0.9)

        # Manually set multi-line text for the fit legend entry
        # Break label into multiple lines for better readability
        fit_label_text = (
            f'Fit: ({p[0]:.2f} ± {err[0]:.2f}) $x^{{3}}$\n'
            f'+ ({p[1]:.2f} ± {err[1]:.2f}) $x^{{2}}$\n'
            f'+ ({p[2]:.2f} ± {err[2]:.2f}) $x$\n'
            f'+ ({p[3]:.2f} ± {err[3]:.2f})'
        )
        # Find the Fit entry and update its text
        for text in legend.get_texts():
            if text.get_text() == 'Fit':
                text.set_text(fit_label_text)

    for ax in axes[len(epochs):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(f"plot_predictions/{filename}.png")
    plt.close(fig)


if __name__ == "__main__":
    # remember to change name each time, might want to automate it somehow
    plot_predictions("silu_bothbcs_hardmode")
