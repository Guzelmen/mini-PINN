import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from scipy.optimize import curve_fit


def plot_pred_phase1(filename):
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


def plot_pred_general(filename: str, enable_fit: bool = False):
    """
    General plotting utility:
    - Loads predictions the same way as plot_pred_phase1 (predictions/{filename}.pkl)
    - Plots y vs x per saved epoch in a grid
    - If enable_fit=False (default): only plots the data points.
      Note: Fitting is currently disabled by request.
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

        sort_idx = np.argsort(inputs)
        x = inputs[sort_idx]
        y = outputs[sort_idx]

        ax.plot(x, y, '+', color="black", label='Model Predictions')

        if enable_fit is True:
            # Candidate functions and initial guesses
            def poly2(x, a0, a1, a2):
                return a0 + a1 * x + a2 * x**2

            def poly3(x, a0, a1, a2, a3):
                return a0 + a1 * x + a2 * x**2 + a3 * x**3

            def poly4(x, a0, a1, a2, a3, a4):
                return a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4

            def exp1(x, a, b, c):
                return a * np.exp(b * x) + c

            def stretch_exp(x, a, b, c, d):
                return a * np.exp(b * np.power(np.clip(x, 1e-8, None), c)) + d

            def power1(x, a, b, c):
                return a * np.power(np.clip(x, 1e-8, None), b) + c

            candidates = {
                "poly2": poly2,
                "poly3": poly3,
                "poly4": poly4,
                "exp": exp1,
                "stretch_exp": stretch_exp,
                "power": power1,
            }

            best_name = None
            best_params = None
            best_cov = None
            best_curve = None
            best_rss = float("inf")

            for name, fn in candidates.items():
                params, cov, *_ = curve_fit(fn, x, y, maxfev=20000)
                y_hat = fn(x, *params)
                rss = float(np.sum((y - y_hat) ** 2))
                is_best = False
                if rss < best_rss:
                    best_rss = rss
                    best_name = name
                    best_params = params
                    best_cov = cov
                    best_curve = y_hat
                    is_best = True
                errs_loop = np.sqrt(np.clip(np.diag(cov), 0.0, None))
                # Debug print for each candidate tried
                print(
                    f"[fit] candidate={name}, rss={rss:.6g}, "
                    f"params={np.array2string(params, precision=4, floatmode='fixed')}, "
                    f"stderr={np.array2string(errs_loop, precision=4, floatmode='fixed')}, "
                    f"best={is_best}"
                )

            if best_curve is not None:
                ax.plot(x, best_curve, color="red", label='Fit')

        ax.set_xlabel('x')
        ax.set_ylabel(r'$\psi(x)$')
        ax.set_title(f"Epoch: {e}")
        legend = ax.legend(loc='best', framealpha=0.9)

        if enable_fit is True:
            if best_curve is not None and best_params is not None and best_cov is not None:
                errs = np.sqrt(np.clip(np.diag(best_cov), 0.0, None))
                # Build a readable multi-line label depending on the chosen model
                if best_name in ("poly2", "poly3", "poly4"):
                    deg = len(best_params) - 1
                    # params are [a0, a1, ..., a_deg]; present from highest power to constant
                    terms = []
                    for k in range(deg, -1, -1):
                        coeff = best_params[k]
                        err = errs[k] if k < len(errs) else 0.0
                        if k >= 2:
                            terms.append(
                                f"({coeff:.3g} ± {err:.2g}) $x^{{{k}}}$")
                        elif k == 1:
                            terms.append(f"({coeff:.3g} ± {err:.2g}) $x$")
                        else:
                            terms.append(f"({coeff:.3g} ± {err:.2g})")
                    fit_label_text = "Fit (poly): " + "\n+ ".join(terms)
                elif best_name == "exp":
                    a, b, c = best_params
                    ea, eb, ec = errs[:3]
                    fit_label_text = (
                        "Fit (exp): $a e^{b x} + c$\n"
                        f"a={a:.3g} ± {ea:.2g}\n"
                        f"b={b:.3g} ± {eb:.2g}\n"
                        f"c={c:.3g} ± {ec:.2g}"
                    )
                elif best_name == "stretch_exp":
                    a, b, c, d = best_params
                    ea, eb, ec, ed = errs[:4]
                    fit_label_text = (
                        "Fit (stretch exp): $a e^{b x^{c}} + d$\n"
                        f"a={a:.3g} ± {ea:.2g}\n"
                        f"b={b:.3g} ± {eb:.2g}\n"
                        f"c={c:.3g} ± {ec:.2g}\n"
                        f"d={d:.3g} ± {ed:.2g}"
                    )
                elif best_name == "power":
                    a, b, c = best_params
                    ea, eb, ec = errs[:3]
                    fit_label_text = (
                        "Fit (power): $a x^{b} + c$\n"
                        f"a={a:.3g} ± {ea:.2g}\n"
                        f"b={b:.3g} ± {eb:.2g}\n"
                        f"c={c:.3g} ± {ec:.2g}"
                    )
                else:
                    # Fallback generic listing
                    parts = []
                    for i, (p, ep) in enumerate(zip(best_params, errs)):
                        parts.append(f"p{i}={p:.3g} ± {ep:.2g}")
                    fit_label_text = "Fit: \n" + "\n".join(parts)

                # Update the legend entry for 'Fit'
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
    plot_pred_general(
        "phase2_hardmode_update_model_coeff_1_300ep", enable_fit=True)
