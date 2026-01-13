import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from scipy.optimize import curve_fit
from .utils import PROJECT_ROOT


def plot_pred_phase1(filepath, params):
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

    Returns: big figure with all plots for specific epochs, saved in png file
    """

    with open(filepath, "rb") as f:
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
    save_path = PROJECT_ROOT / params.plot_dir / f"{params.n_vars}D" / f"{params.run_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_pred_general(filepath: str, params, enable_fit: bool = False):
    """
    General plotting utility:
    - Loads predictions the same way as plot_pred_phase1 (predictions/{filename}.pkl)
    - Plots y vs x per saved epoch in a grid
    - If enable_fit=False (default): only plots the data points.
      Note: Fitting is currently disabled by request.
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    epochs = list(data.keys())
    n_epochs = len(epochs)
    cols = math.ceil(math.sqrt(n_epochs * 1.5))
    rows = math.ceil(n_epochs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for ax, e in zip(axes, epochs):
        inputs = data[e]["x"].flatten()
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
            best_coeffs = None
            best_cov = None
            best_curve = None
            best_rss = float("inf")

            for name, fn in candidates.items():
                coeffs, cov, *_ = curve_fit(fn, x, y, maxfev=20000)
                y_hat = fn(x, *coeffs)
                rss = float(np.sum((y - y_hat) ** 2))
                is_best = False
                if rss < best_rss:
                    best_rss = rss
                    best_name = name
                    best_coeffs = coeffs
                    best_cov = cov
                    best_curve = y_hat
                    is_best = True
                errs_loop = np.sqrt(np.clip(np.diag(cov), 0.0, None))
                # Debug print for each candidate tried
                print(
                    f"[fit] candidate={name}, rss={rss:.6g}, "
                    f"coeffs={np.array2string(coeffs, precision=4, floatmode='fixed')}, "
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
            if best_curve is not None and best_coeffs is not None and best_cov is not None:
                errs = np.sqrt(np.clip(np.diag(best_cov), 0.0, None))
                # Build a readable multi-line label depending on the chosen model
                if best_name in ("poly2", "poly3", "poly4"):
                    deg = len(best_coeffs) - 1
                    # params are [a0, a1, ..., a_deg]; present from highest power to constant
                    terms = []
                    for k in range(deg, -1, -1):
                        coeff = best_coeffs[k]
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
                    a, b, c = best_coeffs
                    ea, eb, ec = errs[:3]
                    fit_label_text = (
                        "Fit (exp): $a e^{b x} + c$\n"
                        f"a={a:.3g} ± {ea:.2g}\n"
                        f"b={b:.3g} ± {eb:.2g}\n"
                        f"c={c:.3g} ± {ec:.2g}"
                    )
                elif best_name == "stretch_exp":
                    a, b, c, d = best_coeffs
                    ea, eb, ec, ed = errs[:4]
                    fit_label_text = (
                        "Fit (stretch exp): $a e^{b x^{c}} + d$\n"
                        f"a={a:.3g} ± {ea:.2g}\n"
                        f"b={b:.3g} ± {eb:.2g}\n"
                        f"c={c:.3g} ± {ec:.2g}\n"
                        f"d={d:.3g} ± {ed:.2g}"
                    )
                elif best_name == "power":
                    a, b, c = best_coeffs
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
                    for i, (p, ep) in enumerate(zip(best_coeffs, errs)):
                        parts.append(f"p{i}={p:.3g} ± {ep:.2g}")
                    fit_label_text = "Fit: \n" + "\n".join(parts)

                # Update the legend entry for 'Fit'
                for text in legend.get_texts():
                    if text.get_text() == 'Fit':
                        text.set_text(fit_label_text)

    for ax in axes[len(epochs):]:
        ax.set_visible(False)

    fig.tight_layout()
    save_path = PROJECT_ROOT / params.plot_dir / f"{params.n_vars}D" / f"{params.run_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_pred_specific(filepath: str, func: str, params):
    """
    Fit and plot a specific function over model predictions.
    - filename: same as other plotters (without extension), loads predictions/{filename}.pkl
    - func: one of {"poly2","poly3","poly4","exp","stretch_exp","power"}
    Saves to png file
    """

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    epochs = list(data.keys())
    n_epochs = len(epochs)
    cols = math.ceil(math.sqrt(n_epochs * 1.5))
    rows = math.ceil(n_epochs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    # Define candidate functions (same forms as in plot_pred_general)
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

    mapping = {
        "poly2": poly2,
        "poly3": poly3,
        "poly4": poly4,
        "exp": exp1,
        "stretch_exp": stretch_exp,
        "power": power1,
    }

    if func not in mapping:
        raise ValueError(
            f"Unknown func '{func}'. Expected one of {list(mapping.keys())}.")

    chosen_fn = mapping[func]

    best_coeffs = None
    best_cov = None

    for ax, e in zip(axes, epochs):
        inputs = data[e]["x"].flatten()
        outputs = data[e]["outputs"].flatten()

        sort_idx = np.argsort(inputs)
        x = inputs[sort_idx]
        y = outputs[sort_idx]

        ax.plot(x, y, '+', color="black", label='Model Predictions')

        coeffs, cov, *_ = curve_fit(chosen_fn, x, y, maxfev=20000)
        y_hat = chosen_fn(x, *coeffs)
        ax.plot(x, y_hat, color="red", label='Fit')

        # Store last params/cov for legend formatting; legend shows for each subplot anyway
        best_coeffs = coeffs
        best_cov = cov

        ax.set_xlabel('x')
        ax.set_ylabel(r'$\psi(x)$')
        ax.set_title(f"Epoch: {e}")
        legend = ax.legend(loc='best', framealpha=0.9)

        # Legend details with parameter uncertainties
        if best_coeffs is not None and best_cov is not None:
            errs = np.sqrt(np.clip(np.diag(best_cov), 0.0, None))
            if func in ("poly2", "poly3", "poly4"):
                deg = len(best_coeffs) - 1
                terms = []
                for k in range(deg, -1, -1):
                    coeff = best_coeffs[k]
                    err = errs[k] if k < len(errs) else 0.0
                    if k >= 2:
                        terms.append(f"({coeff:.3g} ± {err:.2g}) $x^{{{k}}}$")
                    elif k == 1:
                        terms.append(f"({coeff:.3g} ± {err:.2g}) $x$")
                    else:
                        terms.append(f"({coeff:.3g} ± {err:.2g})")
                fit_label_text = f"Fit ({func}): " + "\n+ ".join(terms)
            elif func == "exp":
                a, b, c = best_coeffs
                ea, eb, ec = errs[:3]
                fit_label_text = (
                    "Fit (exp): $a e^{b x} + c$\n"
                    f"a={a:.3g} ± {ea:.2g}\n"
                    f"b={b:.3g} ± {eb:.2g}\n"
                    f"c={c:.3g} ± {ec:.2g}"
                )
            elif func == "stretch_exp":
                a, b, c, d = best_coeffs
                ea, eb, ec, ed = errs[:4]
                fit_label_text = (
                    "Fit (stretch exp): $a e^{b x^{c}} + d$\n"
                    f"a={a:.3g} ± {ea:.2g}\n"
                    f"b={b:.3g} ± {eb:.2g}\n"
                    f"c={c:.3g} ± {ec:.2g}\n"
                    f"d={d:.3g} ± {ed:.2g}"
                )
            elif func == "power":
                a, b, c = best_coeffs
                ea, eb, ec = errs[:3]
                fit_label_text = (
                    "Fit (power): $a x^{b} + c$\n"
                    f"a={a:.3g} ± {ea:.2g}\n"
                    f"b={b:.3g} ± {eb:.2g}\n"
                    f"c={c:.3g} ± {ec:.2g}"
                )
            else:
                parts = []
                for i, (p, ep) in enumerate(zip(best_coeffs, errs)):
                    parts.append(f"p{i}={p:.3g} ± {ep:.2g}")
                fit_label_text = f"Fit ({func}): \n" + "\n".join(parts)

            for text in legend.get_texts():
                if text.get_text() == 'Fit':
                    text.set_text(fit_label_text)

    for ax in axes[len(epochs):]:
        ax.set_visible(False)

    fig.tight_layout()
    save_path = PROJECT_ROOT / params.plot_dir / f"{params.n_vars}D" / f"{params.run_name}_{func}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_pred_fmt_power(filepath: str, params, number=3):
    """
    Fit and plot a specific function over model predictions.
    - filename: same as other plotters (without extension), loads predictions/{filename}.pkl
    - number: number of coefficients in the Feynman-Metropolis-Teller power series 
              approximation to do, default is 3.

    Saves to png file
    """

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    epochs = list(data.keys())
    n_epochs = len(epochs)
    cols = math.ceil(math.sqrt(n_epochs * 1.5))
    rows = math.ceil(n_epochs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), dpi=400)
    axes = axes.flatten()

    def series(x, a2):
        coeff_exp = np.array([
            [a2, 1],
            [4/3, 1.5],
            [0, 2],
            [2/5*a2, 2.5],
            [1/3, 3],
            [3/70*a2, 3.5],
            [2/15*a2, 4],
            [(2/27 - 1/252 * a2**3), 4.5],
            [(1/175 * a2**2), 5],
            [(31/1485 * a2 + 1/1056 * a2**4), 5.5]], dtype=float)

        y = np.ones_like(x, dtype=float)
        k = min(number, len(coeff_exp))
        for i in range(k):
            y += coeff_exp[i][0] * x**coeff_exp[i][1]

        return y

    best_coeffs = None
    best_cov = None

    for ax, e in zip(axes, epochs):
        inputs = data[e]["x"].flatten()
        outputs = data[e]["outputs"].flatten()

        sort_idx = np.argsort(inputs)
        x = inputs[sort_idx]
        y = outputs[sort_idx]

        ax.plot(x, y, '+', color="black", ms=0.5, label='Model Predictions')

        coeffs, cov, *_ = curve_fit(series, x, y, maxfev=20000)
        y_hat = series(x, *coeffs)
        ax.plot(x, y_hat, color="red", label='Fit')

        # Store last params/cov for legend formatting; legend shows for each subplot anyway
        best_coeffs = coeffs
        best_cov = cov

        ax.set_xlabel('x')
        ax.set_ylabel(r'$\psi(x)$')
        ax.set_title(f"Epoch: {e}")
        # legend = ax.legend(loc='best', framealpha=0.9)

        # Legend details with parameter uncertainties
        if best_coeffs is not None and best_cov is not None:
            # a2 and its standard error
            a2 = float(best_coeffs[0])
            coeff_exp = np.array([
                [a2, 1],
                [4/3, 1.5],
                [0, 2],
                [2/5*a2, 2.5],
                [1/3, 3],
                [3/70*a2, 3.5],
                [2/15*a2, 4],
                [(2/27 - 1/252 * a2**3), 4.5],
                [(1/175 * a2**2), 5],
                [(31/1485 * a2 + 1/1056 * a2**4), 5.5]], dtype=float)

            err_a2 = float(np.sqrt(max(best_cov[0, 0], 0.0)))
            # print uncertainty to terminal
            print(f"Epoch {e}: a2 = {a2:.6g} +/- {err_a2:.6g}")

            fit_label_text = "1 "
            k = min(number, len(coeff_exp))
            for i in range(k):
                co = coeff_exp[i][0]
                exp = coeff_exp[i][1]
                exp_val = float(exp)
                if abs(exp_val - round(exp_val)) < 1e-12:
                    int_exp = int(round(exp_val))
                    if int_exp == 1:
                        power_str = "x"
                    else:
                        power_str = f"x^{{{int_exp}}}"
                else:
                    power_str = f"x^{{{exp_val:g}}}"
                fit_label_text = fit_label_text + f"+ {co:.3g} ${power_str}$\n"

            # for text in legend.get_texts():
            #    if text.get_text() == 'Fit':
            #        text.set_text(fit_label_text)

    for ax in axes[len(epochs):]:
        ax.set_visible(False)

    fig.tight_layout()
    save_path = PROJECT_ROOT / params.plot_dir / f"{params.n_vars}D" / f"{params.run_name}_fmt_power_deg{k}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def simple_plot():
    filepath = PROJECT_ROOT / "predictions/2D/addingphi0_training.pkl"
    with open(filepath, "rb") as f:
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
        #d2check = data[e]["d2_check"].flatten()

        sort_idx = np.argsort(inputs)
        x = inputs[sort_idx]
        y1 = outputs[sort_idx]
        #y2 = d2check[sort_idx]
        # Use logarithmic scale for x-axis (x is log-spaced)
        #ax.set_xscale('log')
        #ax.set_xlim(1e-12, 1.0)

        ax.plot(x, y1, '+', color="red", ms=0.5, label=r'$\psi(x)$')
        #ax.plot(x, y2, '+', color="red", ms=0.5, label="d2 term check")

        ax.set_xlabel('x')
        ax.set_title(f"Epoch: {e}")
        legend = ax.legend(loc='best', framealpha=0.9)

    for ax in axes[len(epochs):]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()
    save_path = PROJECT_ROOT / "plot_predictions/1D/300_coeff1_only_pred.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)



def plot_pred_only(filepath: str, params, enable_fit: bool = False, output_dir: str = None):
    """
    General plotting utility:
    - Loads predictions the same way as plot_pred_phase1 (predictions/{filename}.pkl)
    - Plots y vs x per saved epoch in a grid
    - If enable_fit=False (default): only plots the data points.
      Note: Fitting is currently disabled by request.
    - output_dir: Optional custom output directory (relative to PROJECT_ROOT). 
                  If None, uses params.plot_dir / f"{params.n_vars}D"
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    epochs = list(data.keys())
    n_epochs = len(epochs)
    cols = math.ceil(math.sqrt(n_epochs * 1.5))
    rows = math.ceil(n_epochs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for ax, e in zip(axes, epochs):
        inputs = data[e]["x"].flatten()
        outputs = data[e]["outputs"].flatten()
        d2check = data[e]["d2_check"].flatten()

        sort_idx = np.argsort(inputs)
        x = inputs[sort_idx]
        y1 = outputs[sort_idx]
        y2 = d2check[sort_idx]
        # Use logarithmic scale for x-axis (x is log-spaced)
        x_type = getattr(params, "x_type", "normal")
        if x_type == "log":
            ax.set_xscale('log')
            ax.set_xlim(1e-12, 1.0)

        ax.plot(x, y1, '+', color="blue", ms=0.5, label=r'$\psi(x)$')
        #ax.plot(x, y2, '+', color="red", ms=0.5, label="d2 term check")

        if x_type == "log":
            ax.set_xlabel('x (log scale)')
        else:
            ax.set_xlabel("x")
        ax.set_title(f"Epoch: {e}")
        legend = ax.legend(loc='best', framealpha=0.9)

    for ax in axes[len(epochs):]:
        ax.set_visible(False)

    fig.tight_layout()
    
    # Use custom output_dir if provided, otherwise use default path
    if output_dir is not None:
        save_path = PROJECT_ROOT / output_dir / f"{params.run_name}.png"
    else:
        save_path = PROJECT_ROOT / params.plot_dir / f"{params.n_vars}D" / f"{params.run_name}.png"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def simple_log_plot():
    filepath = PROJECT_ROOT / "predictions/2D/standardizing.pkl"
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    epochs = list(data.keys())
    n_epochs = len(epochs)
    cols = math.ceil(math.sqrt(n_epochs * 1.5))
    rows = math.ceil(n_epochs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for ax, e in zip(axes, epochs):
        x_vals = data[e]["x"].flatten()
        outputs = data[e]["outputs"].flatten()

        sort_idx = np.argsort(x_vals)
        x = x_vals[sort_idx]
        y1 = outputs[sort_idx]

        # Log-log plot: both axes on logarithmic scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-4, 1.0)
        ax.plot(x, y1, '+', color="red", ms=0.5, label=r'$\psi(x)$')

        ax.set_xlabel('x (log scale)')
        ax.set_ylabel(r'$\psi(x)$ (log scale)')
        ax.set_title(f"Epoch: {e}")
        legend = ax.legend(loc='best', framealpha=0.9)

    for ax in axes[len(epochs):]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()
    save_path = PROJECT_ROOT / "plot_predictions/2D/standardizing_log.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def simple_plot_latest_epoch_only():
    filename = "addingphi0_training"
    filepath = PROJECT_ROOT / f"predictions/2D/{filename}.pkl"
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    # Get the latest epoch (last key, relying on insertion order)
    epochs = list(data.keys())
    if not epochs:
        raise ValueError("No epochs found in predictions file.")
    latest_epoch = epochs[-1]

    fig, ax = plt.subplots(figsize=(6, 4))

    inputs = data[latest_epoch]["x"].flatten()
    outputs = data[latest_epoch]["outputs"].flatten()

    sort_idx = np.argsort(inputs)
    x = inputs[sort_idx]
    y1 = outputs[sort_idx]

    ax.plot(x, y1, '+', color="red", ms=0.5, label=r'$\psi(x)$')

    ax.set_xlabel('x')
    ax.set_title(f"Epoch: {latest_epoch}")
    ax.legend(loc='best', framealpha=0.9)

    fig.tight_layout()
    plt.show()
    save_path = PROJECT_ROOT / f"plot_predictions/2D/{filename}_latestepoch.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    simple_plot_latest_epoch_only()
