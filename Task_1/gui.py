import os
import io
import base64
import flet as ft
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import LoadingData
from perceptron_model import PerceptronModel
from adaline_model import AdalineModel
from nn_metrics import conf_mat, acc

# ---------- Colors ----------
BG = "#1e1e2e"
SIDEBAR = "#2a2a3c"
CARD = "#1a1a28"
ACCENT = "#7aa2f7"
TEXT = "#e0e0e0"
SUBTLE = "#565f89"
GREEN = "#9ece6a"
RED = "#f7768e"
ERROR_BG = "#3a1e28"
SUCCESS_BG = "#1e3a24"
GOLD = "#e0af68"


def main(page: ft.Page):
    page.title = "Penguins Classifier — Perceptron vs Adaline"
    page.bgcolor = BG
    page.padding = 0
    page.window_width = 1280  # type: ignore
    page.window_height = 800  # type: ignore
    page.window_min_width = 1000  # type: ignore
    page.window_min_height = 650  # type: ignore

    # ---------- Paths ----------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---------- Load Data ----------
    csv_path = os.path.join(BASE_DIR, "..", "penguins.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, "penguins.csv")

    loader = LoadingData(csv_path)

    # ---------- State ----------
    state = {
        "model": None,
        "X_train": None,
        "y_train": None,
        "X_test": None,
        "y_test": None,
        "X_train_raw": None,
        "X_test_raw": None,
        "mean": None,
        "std": None,
        "w": None,
        "b": 0.0,
    }

    # ---------- Results Log ----------
    results_log = []  # one dict per training run

    # =========================================================
    # HELPERS
    # =========================================================
    def label(txt, size=13, bold=False, color=TEXT):
        return ft.Text(
            txt,
            size=size,
            color=color,
            weight=ft.FontWeight.BOLD if bold else ft.FontWeight.NORMAL,
        )

    def _input_border(color=SUBTLE):
        return ft.OutlineInputBorder(
            side=ft.BorderSide(1, color),
            border_radius=8,
        )

    def _style_axes(ax):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(SUBTLE)
        ax.set_xlabel("Feature X", color=TEXT, fontsize=10)
        ax.set_ylabel("Feature Y", color=TEXT, fontsize=10)
        ax.set_title(
            "Train a model to see decision boundary", color=SUBTLE, fontsize=10
        )

    # =========================================================
    # FIGURE → BASE64 PNG
    # =========================================================
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=100, facecolor=CARD)
    _style_axes(ax)

    def fig_to_base64(f):
        buf = io.BytesIO()
        f.savefig(buf, format="png", facecolor=CARD, bbox_inches="tight", dpi=100)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    plot_image = ft.Image(
        src="",
        expand=True,
        fit=ft.BoxFit.CONTAIN,
    )
    plot_image.src = f"data:image/png;base64,{fig_to_base64(fig)}"

    plot_container = ft.Container(
        content=plot_image,
        bgcolor=CARD,
        border_radius=10,
        padding=10,
        expand=True,
    )

    # =========================================================
    # UNIFIED BANNER
    # =========================================================
    banner_text = ft.Text("", size=13, selectable=True, expand=True, color=RED)
    banner_icon = ft.Icon(ft.Icons.ERROR_OUTLINE, color=RED, size=20)

    banner = ft.Container(
        content=ft.Row(
            controls=[
                banner_icon,
                banner_text,
                ft.IconButton(
                    icon=ft.Icons.CLOSE,
                    icon_color=RED,
                    icon_size=18,
                    tooltip="Dismiss",
                    on_click=lambda e: hide_banner(),
                ),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        bgcolor=ERROR_BG,
        border=ft.Border.all(1, RED),
        border_radius=8,
        padding=ft.Padding.symmetric(horizontal=12, vertical=8),
        visible=False,
        margin=ft.Margin.only(bottom=8),
    )

    def show_error(msg: str):
        banner_text.value = msg
        banner_text.color = RED
        banner_icon.name = ft.Icons.ERROR_OUTLINE  # type: ignore
        banner_icon.color = RED
        banner.bgcolor = ERROR_BG
        banner.border = ft.Border.all(1, RED)
        banner.content.controls[-1].icon_color = RED  # type: ignore
        banner.visible = True
        page.update()

    def show_success(msg: str):
        banner_text.value = msg
        banner_text.color = GREEN
        banner_icon.name = ft.Icons.CHECK_CIRCLE_OUTLINE  # type: ignore
        banner_icon.color = GREEN
        banner.bgcolor = SUCCESS_BG
        banner.border = ft.Border.all(1, GREEN)
        banner.content.controls[-1].icon_color = GREEN  # type: ignore
        banner.visible = True
        page.update()

    def hide_banner():
        banner.visible = False
        page.update()

    # =========================================================
    # SIDEBAR — Inputs
    # =========================================================
    def dropdown(label_text, options, width=160):
        return ft.Column(
            spacing=4,
            controls=[
                label(label_text, size=12, color=SUBTLE),
                ft.Dropdown(
                    value=None,
                    width=width,
                    options=[ft.dropdown.Option(o) for o in options],
                    text_style=ft.TextStyle(color=TEXT, size=13),
                    bgcolor="#3a3a50",
                    border=_input_border(),
                    content_padding=10,
                    hint_text="Select...",
                    hint_style=ft.TextStyle(color=SUBTLE, size=12),
                ),
            ],
        )

    feat_x = dropdown("Feature X", loader.features_name)
    feat_y = dropdown("Feature Y", loader.features_name)
    class_a = dropdown("Class A (+1)", loader.species_list)
    class_b = dropdown("Class B (-1)", loader.species_list)

    eta_input = ft.TextField(
        value="",
        width=160,
        bgcolor="#3a3a50",
        color=TEXT,
        hint_text="e.g. 0.01",
        hint_style=ft.TextStyle(color=SUBTLE, size=12),
        border=_input_border(),
        text_size=13,
        content_padding=10,
    )
    epochs_input = ft.TextField(
        value="",
        width=160,
        bgcolor="#3a3a50",
        color=TEXT,
        hint_text="e.g. 500",
        hint_style=ft.TextStyle(color=SUBTLE, size=12),
        border=_input_border(),
        text_size=13,
        content_padding=10,
    )
    mse_input = ft.TextField(
        value="",
        width=160,
        bgcolor="#3a3a50",
        color=TEXT,
        hint_text="e.g. 0.001",
        hint_style=ft.TextStyle(color=SUBTLE, size=12),
        border=_input_border(),
        text_size=13,
        content_padding=10,
    )

    bias_check = ft.Checkbox(
        label="Use Bias",
        value=False,
        label_style=ft.TextStyle(color=TEXT, size=13),
        fill_color=ACCENT,
        check_color=BG,
    )

    algo_radio = ft.RadioGroup(
        value=None,
        content=ft.Column(
            [
                ft.Radio(
                    value="Perceptron",
                    label="Perceptron",
                    label_style=ft.TextStyle(color=TEXT, size=13),
                    fill_color=ACCENT,
                ),
                ft.Radio(
                    value="Adaline",
                    label="Adaline",
                    label_style=ft.TextStyle(color=TEXT, size=13),
                    fill_color=ACCENT,
                ),
            ],
            spacing=2,
        ),
    )

    status_text = ft.Text(
        "Ready.", size=11, color=SUBTLE, italic=True, width=260, no_wrap=False
    )

    sidebar = ft.Container(
        width=310,
        bgcolor=SIDEBAR,
        padding=20,
        content=ft.Column(
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
            controls=[
                label("Configuration", size=18, bold=True, color=ACCENT),
                feat_x,
                feat_y,
                ft.Divider(color=SUBTLE),
                class_a,
                class_b,
                ft.Divider(color=SUBTLE),
                label("Learning Rate (η):", size=12, color=SUBTLE),
                eta_input,
                label("Epochs:", size=12, color=SUBTLE),
                epochs_input,
                label("MSE Threshold:", size=12, color=SUBTLE),
                mse_input,
                bias_check,
                ft.Divider(color=SUBTLE),
                label("Algorithm", size=15, bold=True, color=ACCENT),
                algo_radio,
                ft.Divider(color=SUBTLE),
                ft.Button(
                    content="▶  Train & Evaluate",
                    width=260,
                    style=ft.ButtonStyle(
                        bgcolor=ACCENT,
                        color=BG,
                        padding=14,
                        shape=ft.RoundedRectangleBorder(radius=8),
                        text_style=ft.TextStyle(size=14, weight=ft.FontWeight.BOLD),
                    ),
                    on_click=lambda e: on_train(e),
                ),
                ft.Button(
                    content="💾  Save Plot",
                    width=260,
                    style=ft.ButtonStyle(
                        bgcolor="#3a3a50",
                        color=GREEN,
                        padding=12,
                        shape=ft.RoundedRectangleBorder(radius=8),
                        text_style=ft.TextStyle(size=13, weight=ft.FontWeight.BOLD),
                    ),
                    on_click=lambda e: on_save_plot(e),
                ),
                ft.Button(
                    content="📈  MSE History",
                    width=260,
                    style=ft.ButtonStyle(
                        bgcolor="#3a3a50",
                        color=ACCENT,
                        padding=12,
                        shape=ft.RoundedRectangleBorder(radius=8),
                        text_style=ft.TextStyle(size=13, weight=ft.FontWeight.BOLD),
                    ),
                    on_click=lambda e: on_show_mse(e),
                ),
                ft.Button(
                    content="📊  Results Log",
                    width=260,
                    style=ft.ButtonStyle(
                        bgcolor="#3a3a50",
                        color=GOLD,
                        padding=12,
                        shape=ft.RoundedRectangleBorder(radius=8),
                        text_style=ft.TextStyle(size=13, weight=ft.FontWeight.BOLD),
                    ),
                    on_click=lambda e: on_show_results(e),
                ),
                ft.OutlinedButton(
                    content="Reset",
                    width=260,
                    style=ft.ButtonStyle(
                        color=TEXT,
                        side=ft.BorderSide(1, SUBTLE),
                        padding=12,
                        shape=ft.RoundedRectangleBorder(radius=8),
                    ),
                    on_click=lambda e: on_reset(e),
                ),
                ft.Divider(color=SUBTLE),
                status_text,
            ],
        ),
    )

    # =========================================================
    # RESULTS PANEL
    # =========================================================
    cm_text = ft.Text(
        "Confusion Matrix:\n—",
        color=TEXT,
        size=12,
        font_family="Consolas",
        no_wrap=False,
    )
    acc_text = ft.Text("Accuracy: —", color=GREEN, size=15, weight=ft.FontWeight.BOLD)
    mse_text = ft.Text("Train MSE: —", color=TEXT, size=12)
    val_mse_text = ft.Text("Val MSE: —", color=TEXT, size=12)
    epochs_text = ft.Text("Epochs run: —", color=TEXT, size=12)
    w_text = ft.Text("w = (—, —)\nb = —", color=TEXT, size=12, font_family="Consolas")
    norm_note = ft.Text(
        "Features normalized (z-score).", color=SUBTLE, size=10, italic=True
    )

    sample_x = ft.TextField(
        label="x",
        width=80,
        bgcolor="#3a3a50",
        color=TEXT,
        border=_input_border(),
        text_size=12,
        content_padding=8,
    )
    sample_y = ft.TextField(
        label="y",
        width=80,
        bgcolor="#3a3a50",
        color=TEXT,
        border=_input_border(),
        text_size=12,
        content_padding=8,
    )
    sample_result = ft.Text(
        "Prediction: —", color=RED, size=13, weight=ft.FontWeight.BOLD
    )

    results_panel = ft.Container(
        width=280,
        bgcolor=CARD,
        border_radius=10,
        padding=16,
        content=ft.Column(
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
            controls=[
                label("Evaluation", size=16, bold=True, color=ACCENT),
                cm_text,
                acc_text,
                mse_text,
                val_mse_text,
                epochs_text,
                ft.Divider(color=SUBTLE),
                label("Weights", size=16, bold=True, color=ACCENT),
                w_text,
                norm_note,
                ft.Divider(color=SUBTLE),
                label("Classify a Sample", size=16, bold=True, color=ACCENT),
                label("(enter values in original scale)", size=10, color=SUBTLE),
                ft.Row([sample_x, sample_y], spacing=8),
                ft.Button(
                    content="Classify",
                    style=ft.ButtonStyle(
                        bgcolor=ACCENT,
                        color=BG,
                        padding=10,
                        shape=ft.RoundedRectangleBorder(radius=6),
                    ),
                    on_click=lambda e: on_classify_sample(e),
                ),
                sample_result,
            ],
        ),
    )

    main_area = ft.Row(
        expand=True,
        spacing=14,
        controls=[plot_container, results_panel],
    )

    right_side = ft.Container(
        expand=True,
        padding=ft.Padding.only(left=16, right=16, top=16, bottom=16),
        content=ft.Column(
            expand=True,
            controls=[
                label("Decision Boundary Visualizer", size=22, bold=True),
                ft.Container(height=8),
                banner,
                main_area,
            ],
        ),
    )

    page.add(
        ft.Row(
            expand=True,
            spacing=0,
            controls=[sidebar, right_side],
        )
    )

    # =========================================================
    # EVENT HANDLERS
    # =========================================================
    def on_train(e):
        hide_banner()
        try:
            fx = feat_x.controls[1].value  # type: ignore
            fy = feat_y.controls[1].value  # type: ignore
            ca = class_a.controls[1].value  # type: ignore
            cb = class_b.controls[1].value  # type: ignore
            algo = algo_radio.value

            if not fx or not fy:
                raise ValueError("Please select both Feature X and Feature Y.")
            if not ca or not cb:
                raise ValueError("Please select both Class A and Class B.")
            if not algo:
                raise ValueError("Please choose an algorithm (Perceptron or Adaline).")
            if not eta_input.value:
                raise ValueError("Please enter a learning rate (η).")
            if not epochs_input.value:
                raise ValueError("Please enter the number of epochs.")
            if not mse_input.value:
                raise ValueError("Please enter an MSE threshold.")

            try:
                eta = float(eta_input.value)
            except (ValueError, TypeError):
                raise ValueError("Learning rate (η) must be a valid number.")
            try:
                epochs = int(epochs_input.value)
            except (ValueError, TypeError):
                raise ValueError("Epochs must be a valid integer.")
            try:
                mse_thr = float(mse_input.value)
            except (ValueError, TypeError):
                raise ValueError("MSE threshold must be a valid number.")

            use_bias = bool(bias_check.value)

            if fx == fy:
                raise ValueError("Feature X and Feature Y must be different.")
            if ca == cb:
                raise ValueError("Class A and Class B must be different.")
            if eta <= 0:
                raise ValueError("Learning rate (η) must be > 0.")
            if epochs <= 0:
                raise ValueError("Epochs must be a positive integer.")
            if mse_thr < 0:
                raise ValueError("MSE threshold must be ≥ 0.")

            status_text.value = "Loading & splitting data..."
            page.update()

            data = loader.get_data(fx, fy, ca, cb, normalize=True)

            state["X_train"] = data["X_train"]
            state["y_train"] = data["y_train"]
            state["X_test"] = data["X_test"]
            state["y_test"] = data["y_test"]
            state["X_train_raw"] = data["X_train_raw"]
            state["X_test_raw"] = data["X_test_raw"]
            state["mean"] = data["mean"]
            state["std"] = data["std"]

            status_text.value = f"Training {algo}..."
            page.update()

            if algo == "Perceptron":
                model = PerceptronModel(
                    eta=eta,
                    epochs=epochs,
                    use_bias=use_bias,
                    mse_threshold=mse_thr,
                    random_state=42,
                )
            else:
                model = AdalineModel(
                    eta=eta,
                    epochs=epochs,
                    use_bias=use_bias,
                    mse_threshold=mse_thr,
                    random_state=42,
                )

            model.fitting(  # type: ignore
                state["X_train"],
                state["y_train"],
                X_val=state["X_test"],
                y_val=state["y_test"],
            )

            state["model"] = model
            state["w"] = model.w
            state["b"] = model.b if use_bias else 0.0

            y_pred = model.predict(state["X_test"])
            cm = conf_mat(state["y_test"], y_pred)
            ac = acc(state["y_test"], y_pred)

            _update_plot(fx, fy, ca, cb)
            _update_results(cm, ac, model)

            status_text.value = f"{algo} finished. Epochs run: {model.epochs_run}"
            page.update()
            show_success(f"{algo} training finished — Accuracy: {ac*100:.1f}%")

            # ---- Log this run ----
            results_log.append(
                {
                    "algorithm": algo,
                    "feature_x": fx,
                    "feature_y": fy,
                    "class_a": ca,
                    "class_b": cb,
                    "eta": eta,
                    "epochs_max": epochs,
                    "use_bias": use_bias,
                    "accuracy": ac * 100,
                    "train_mse": model.final_mse,
                    "val_mse": (
                        model.val_mse_history[-1]
                        if getattr(model, "val_mse_history", None)
                        else None
                    ),
                    "epochs_run": model.epochs_run,
                }
            )

        except ValueError as ve:
            show_error(f"Input error: {ve}")
            status_text.value = "Error. See banner."
            page.update()
        except FileNotFoundError as fe:
            show_error(f"Data file not found: {fe}")
            status_text.value = "Error. See banner."
            page.update()
        except Exception as ex:
            show_error(f"Unexpected error: {type(ex).__name__}: {ex}")
            status_text.value = "Error. See banner."
            page.update()

    def _update_plot(fx, fy, ca, cb):
        ax.clear()
        _style_axes(ax)

        Xtr_raw = state["X_train_raw"]
        ytr = state["y_train"]
        Xte_raw = state["X_test_raw"]
        yte = state["y_test"]

        # --- Train data ---
        ax.scatter(
            Xtr_raw[ytr == 1, 0],
            Xtr_raw[ytr == 1, 1],
            c=ACCENT,
            edgecolors="white",
            s=45,
            label=f"{ca} train (+1)",
        )
        ax.scatter(
            Xtr_raw[ytr == -1, 0],
            Xtr_raw[ytr == -1, 1],
            c=RED,
            edgecolors="white",
            s=45,
            label=f"{cb} train (-1)",
        )

        # --- Test data ---
        ax.scatter(
            Xte_raw[yte == 1, 0],
            Xte_raw[yte == 1, 1],
            facecolors="none",
            edgecolors=ACCENT,
            s=70,
            marker="s",
            linewidths=1.8,
            label=f"{ca} test (+1)",
        )
        ax.scatter(
            Xte_raw[yte == -1, 0],
            Xte_raw[yte == -1, 1],
            facecolors="none",
            edgecolors=RED,
            s=70,
            marker="s",
            linewidths=1.8,
            label=f"{cb} test (-1)",
        )

        # --- Data ranges (raw) ---
        x_all = np.concatenate([Xtr_raw[:, 0], Xte_raw[:, 0]])
        y_all = np.concatenate([Xtr_raw[:, 1], Xte_raw[:, 1]])
        x_min_r, x_max_r = x_all.min(), x_all.max()
        y_min_r, y_max_r = y_all.min(), y_all.max()
        pad_x = (x_max_r - x_min_r) * 0.05
        pad_y = (y_max_r - y_min_r) * 0.15

        # --- Decision boundary ---
        w1, w2 = state["w"][0], state["w"][1]
        b = state["b"]
        mean = state["mean"]
        std = state["std"]

        x1_raw = x_min_r - pad_x
        x2_raw = x_max_r + pad_x

        if abs(w2) > 1e-9:
            x1_norm = (x1_raw - mean[0]) / std[0]
            x2_norm = (x2_raw - mean[0]) / std[0]
            y1_norm = -(w1 * x1_norm + b) / w2
            y2_norm = -(w1 * x2_norm + b) / w2
            y1_raw = y1_norm * std[1] + mean[1]
            y2_raw = y2_norm * std[1] + mean[1]

            ax.plot(
                [x1_raw, x2_raw],
                [y1_raw, y2_raw],
                color=GREEN,
                linewidth=2,
                label="Decision Boundary",
            )
            ax.scatter(
                [x1_raw, x2_raw],
                [y1_raw, y2_raw],
                c=GREEN,
                s=70,
                marker="*",
                zorder=5,
                edgecolors="white",
                linewidths=1,
                label="Boundary points",
            )

        elif abs(w1) > 1e-9:
            xv_norm = -b / w1
            xv_raw = xv_norm * std[0] + mean[0]
            ax.axvline(xv_raw, color=GREEN, linewidth=2, label="Decision Boundary")
            ax.scatter(
                [xv_raw, xv_raw],
                [y_min_r, y_max_r],
                c=GREEN,
                s=70,
                marker="*",
                zorder=5,
                edgecolors="white",
                linewidths=1,
                label="Boundary points",
            )

        ax.set_xlim(x_min_r - pad_x, x_max_r + pad_x)
        ax.set_ylim(y_min_r - pad_y, y_max_r + pad_y)

        ax.set_xlabel(fx, color=TEXT, fontsize=10)
        ax.set_ylabel(fy, color=TEXT, fontsize=10)
        ax.set_title(f"{ca} vs {cb}", color=TEXT, fontsize=12, fontweight="bold")
        leg = ax.legend(
            facecolor=CARD, edgecolor=SUBTLE, labelcolor=TEXT, fontsize=8, loc="best"
        )
        if leg:
            leg.get_frame().set_alpha(0.9)

        plot_image.src = f"data:image/png;base64,{fig_to_base64(fig)}"
        plot_image.update()

    def _update_results(cm, acc, model):
        TN, FP = cm[0]
        FN, TP = cm[1]
        cm_text.value = (
            f"Confusion Matrix:\n"
            f"  TN = {TN:<4}  FP = {FP:<4}\n"
            f"  FN = {FN:<4}  TP = {TP:<4}"
        )
        acc_text.value = f"Accuracy: {acc * 100:.2f}%"

        train_mse = model.final_mse if model.final_mse is not None else 0.0
        mse_text.value = f"Train MSE: {train_mse:.6f}"

        if getattr(model, "val_mse_history", None):
            val_mse_text.value = f"Val MSE: {model.val_mse_history[-1]:.6f}"
        else:
            val_mse_text.value = "Val MSE: —"

        epochs_text.value = f"Epochs run: {model.epochs_run}"
        w_text.value = (
            f"w = ({state['w'][0]:.4f}, {state['w'][1]:.4f})\n" f"b = {state['b']:.4f}"
        )

    def on_classify_sample(e):
        hide_banner()
        if state["model"] is None:
            show_error("Please train a model first before classifying.")
            return
        try:
            if not sample_x.value or not sample_y.value:
                raise ValueError("Please fill in both x and y values.")
            x_raw = float(sample_x.value)
            y_raw = float(sample_y.value)

            mean = state["mean"]
            std = state["std"]
            x_norm = (x_raw - mean[0]) / std[0]
            y_norm = (y_raw - mean[1]) / std[1]

            sample = np.array([[x_norm, y_norm]])
            pred = state["model"].predict(sample)[0]

            ca = class_a.controls[1].value  # type: ignore
            cb = class_b.controls[1].value  # type: ignore
            cls = ca if pred == 1 else cb
            sample_result.value = f"Prediction: {cls} ({pred:+d})"
        except ValueError as ve:
            show_error(f"Invalid input: {ve}")
            sample_result.value = "Prediction: —"
        except Exception as ex:
            show_error(f"Unexpected error: {type(ex).__name__}: {ex}")
            sample_result.value = "Prediction: —"
        page.update()

    # ---------- Save Plot (auto-save to ./plots/) ----------
    def on_save_plot(e):
        hide_banner()

        if state["model"] is None:
            show_error("Please train a model first before saving the plot.")
            return

        try:
            fx = feat_x.controls[1].value  # type: ignore
            fy = feat_y.controls[1].value  # type: ignore
            ca = class_a.controls[1].value  # type: ignore
            cb = class_b.controls[1].value  # type: ignore
            algo = algo_radio.value
            use_bias = bias_check.value

            bias_tag = "bias" if use_bias else "nobias"
            filename = f"{algo}_{ca}_vs_{cb}_{fx}_{fy}_{bias_tag}.png".replace(" ", "_")
            save_path = os.path.join(PLOTS_DIR, filename)

            fig.savefig(
                save_path, format="png", facecolor=CARD, bbox_inches="tight", dpi=150
            )

            show_success(f"Plot saved: {filename}")
            status_text.value = f"Saved: {filename}"
            page.update()

        except Exception as ex:
            show_error(f"Failed to save plot: {type(ex).__name__}: {ex}")
            page.update()

    # ---------- MSE History ----------
    def on_show_mse(e):
        hide_banner()

        if state["model"] is None:
            show_error("Please train a model first.")
            return

        model = state["model"]
        if not model.mse_history:
            show_error("No MSE history recorded.")
            return

        fig2, ax2 = plt.subplots(figsize=(8, 4.5), dpi=100, facecolor=CARD)
        ax2.set_facecolor(CARD)
        ax2.tick_params(colors=TEXT, labelsize=9)
        for spine in ax2.spines.values():
            spine.set_color(SUBTLE)

        epochs_x = list(range(1, len(model.mse_history) + 1))

        ax2.plot(
            epochs_x, model.mse_history, color=ACCENT, linewidth=2, label="Train MSE"
        )

        if model.val_mse_history:
            ax2.plot(
                epochs_x,
                model.val_mse_history,
                color=GREEN,
                linewidth=2,
                label="Validation MSE",
            )

        ax2.axhline(
            y=model.mse_threshold,
            color=RED,
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({model.mse_threshold})",
        )

        ax2.set_xlabel("Epoch", color=TEXT, fontsize=10)
        ax2.set_ylabel("MSE", color=TEXT, fontsize=10)
        ax2.set_title(
            f"MSE History — {algo_radio.value}",
            color=TEXT,
            fontsize=12,
            fontweight="bold",
        )
        leg = ax2.legend(facecolor=CARD, edgecolor=SUBTLE, labelcolor=TEXT, fontsize=9)
        if leg:
            leg.get_frame().set_alpha(0.9)
        ax2.grid(True, alpha=0.2, color=SUBTLE)

        # Auto-save MSE history image too
        try:
            fx = feat_x.controls[1].value  # type: ignore
            fy = feat_y.controls[1].value  # type: ignore
            ca = class_a.controls[1].value  # type: ignore
            cb = class_b.controls[1].value  # type: ignore
            algo = algo_radio.value
            bias_tag = "bias" if bias_check.value else "nobias"
            mse_name = f"MSE_{algo}_{ca}_vs_{cb}_{fx}_{fy}_{bias_tag}.png".replace(
                " ", "_"
            )
            fig2.savefig(
                os.path.join(PLOTS_DIR, mse_name),
                format="png",
                facecolor=CARD,
                bbox_inches="tight",
                dpi=120,
            )
        except Exception:
            pass  # not critical

        img_src = f"data:image/png;base64,{fig_to_base64(fig2)}"
        plt.close(fig2)

        dialog_image = ft.Image(
            src=img_src,
            fit=ft.BoxFit.CONTAIN,
            width=760,
            height=420,
        )

        def close_dialog(e):
            page.pop_dialog()

        dlg = ft.AlertDialog(
            modal=True,
            bgcolor=CARD,
            title=ft.Text(
                "MSE History", color=ACCENT, size=16, weight=ft.FontWeight.BOLD
            ),
            content=ft.Container(content=dialog_image, bgcolor=CARD),
            actions=[
                ft.TextButton(
                    content=ft.Text("Close", color=TEXT),
                    on_click=close_dialog,
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        page.show_dialog(dlg)

    # =========================================================
    # RESULTS LOG
    # =========================================================
    def _build_results_table():
        if not results_log:
            return ft.Text(
                "No results logged yet. Train a model first.",
                color=SUBTLE,
                size=12,
                italic=True,
            )

        headers = [
            ("#", 30),
            ("Algo", 80),
            ("Feat X", 110),
            ("Feat Y", 110),
            ("Class A", 80),
            ("Class B", 80),
            ("η", 55),
            ("Epochs", 60),
            ("Bias", 50),
            ("Acc %", 60),
            ("Train MSE", 85),
            ("Val MSE", 85),
            ("Ep. Run", 65),
        ]

        columns = [
            ft.DataColumn(
                label=ft.Text(h, color=ACCENT, size=10, weight=ft.FontWeight.BOLD),
            )
            for h, _ in headers
        ]

        rows = []
        for idx, r in enumerate(results_log, start=1):

            def cell(txt, color=TEXT):
                return ft.DataCell(ft.Text(str(txt), color=color, size=10))

            val_mse_str = f"{r['val_mse']:.5f}" if r["val_mse"] is not None else "—"

            rows.append(
                ft.DataRow(
                    cells=[
                        cell(idx),
                        cell(
                            r["algorithm"],
                            color=ACCENT if r["algorithm"] == "Adaline" else GREEN,
                        ),
                        cell(r["feature_x"]),
                        cell(r["feature_y"]),
                        cell(r["class_a"]),
                        cell(r["class_b"]),
                        cell(r["eta"]),
                        cell(r["epochs_max"]),
                        cell("✓" if r["use_bias"] else "✗"),
                        cell(
                            f"{r['accuracy']:.1f}",
                            color=(
                                GREEN
                                if r["accuracy"] >= 90
                                else (RED if r["accuracy"] < 70 else TEXT)
                            ),
                        ),
                        cell(
                            f"{r['train_mse']:.5f}"
                            if r["train_mse"] is not None
                            else "—"
                        ),
                        cell(val_mse_str),
                        cell(r["epochs_run"]),
                    ]
                )
            )

        return ft.DataTable(
            columns=columns,
            rows=rows,
            bgcolor=CARD,
            border=ft.Border.all(1, SUBTLE),
            border_radius=8,
            heading_row_color="#252536",
            heading_row_height=36,
            data_row_min_height=32,
            data_row_max_height=40,
            column_spacing=12,
            horizontal_margin=8,
        )

    def on_show_results(e):
        hide_banner()

        if not results_log:
            show_error("No results logged yet. Train a model first.")
            return

        table = _build_results_table()

        def export_csv(ev):
            try:
                csv_path_out = os.path.join(BASE_DIR, "results_log.csv")
                headers = [
                    "algorithm",
                    "feature_x",
                    "feature_y",
                    "class_a",
                    "class_b",
                    "eta",
                    "epochs_max",
                    "use_bias",
                    "accuracy",
                    "train_mse",
                    "val_mse",
                    "epochs_run",
                ]
                with open(csv_path_out, "w", encoding="utf-8") as f:
                    f.write(",".join(headers) + "\n")
                    for r in results_log:
                        row = [
                            str(r["algorithm"]),
                            str(r["feature_x"]),
                            str(r["feature_y"]),
                            str(r["class_a"]),
                            str(r["class_b"]),
                            str(r["eta"]),
                            str(r["epochs_max"]),
                            str(r["use_bias"]),
                            f"{r['accuracy']:.4f}",
                            (
                                f"{r['train_mse']:.6f}"
                                if r["train_mse"] is not None
                                else ""
                            ),
                            (f"{r['val_mse']:.6f}" if r["val_mse"] is not None else ""),
                            str(r["epochs_run"]),
                        ]
                        f.write(",".join(row) + "\n")

                show_success(f"CSV saved to: {csv_path_out}")
                page.update()
            except Exception as ex:
                show_error(f"Export failed: {type(ex).__name__}: {ex}")
                page.update()

        def clear_log(ev):
            results_log.clear()
            page.pop_dialog()
            show_success("Results log cleared.")
            page.update()

        def close_dialog(ev):
            page.pop_dialog()

        dlg = ft.AlertDialog(
            modal=True,
            bgcolor=CARD,
            title=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.TABLE_CHART, color=GOLD, size=22),
                    ft.Text(
                        f"Results Log ({len(results_log)} runs)",
                        color=ACCENT,
                        size=16,
                        weight=ft.FontWeight.BOLD,
                    ),
                ],
                spacing=8,
            ),
            content=ft.Container(
                content=ft.Column(
                    controls=[table],
                    scroll=ft.ScrollMode.AUTO,
                ),
                width=1050,
                height=420,
                bgcolor=CARD,
                padding=8,
            ),
            actions=[
                ft.Button(
                    content="📥  Export CSV",
                    style=ft.ButtonStyle(
                        bgcolor="#3a3a50",
                        color=GREEN,
                        padding=10,
                        shape=ft.RoundedRectangleBorder(radius=6),
                    ),
                    on_click=export_csv,
                ),
                ft.Button(
                    content="🗑  Clear Log",
                    style=ft.ButtonStyle(
                        bgcolor="#3a3a50",
                        color=RED,
                        padding=10,
                        shape=ft.RoundedRectangleBorder(radius=6),
                    ),
                    on_click=clear_log,
                ),
                ft.TextButton(
                    content=ft.Text("Close", color=TEXT),
                    on_click=close_dialog,
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        page.show_dialog(dlg)

    def on_reset(e):
        hide_banner()

        for k in state:
            state[k] = None
        state["b"] = 0.0

        ax.clear()
        _style_axes(ax)
        ax.set_title(
            "Train a model to see decision boundary", color=SUBTLE, fontsize=10
        )
        plot_image.src = f"data:image/png;base64,{fig_to_base64(fig)}"
        plot_image.update()

        cm_text.value = "Confusion Matrix:\n—"
        acc_text.value = "Accuracy: —"
        mse_text.value = "Train MSE: —"
        val_mse_text.value = "Val MSE: —"
        epochs_text.value = "Epochs run: —"
        w_text.value = "w = (—, —)\nb = —"
        sample_result.value = "Prediction: —"
        status_text.value = "Reset. Ready."

        sample_x.value = ""
        sample_y.value = ""

        feat_x.controls[1].value = None  # type: ignore
        feat_y.controls[1].value = None  # type: ignore
        class_a.controls[1].value = None  # type: ignore
        class_b.controls[1].value = None  # type: ignore
        algo_radio.value = None
        eta_input.value = ""
        epochs_input.value = ""
        mse_input.value = ""
        bias_check.value = False

        page.update()


if __name__ == "__main__":
    ft.run(main)
