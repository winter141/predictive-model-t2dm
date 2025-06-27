from enum import Enum
from typing import Union
import numpy as np
import shap
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from models import get_new_preds, ModelType, SHAP_analysis, split_train_test, xgboost, gradient_boosting, PDP_analysis

R_ITERATIONS = 100


def plt_model_results(preds, y_test, model_name: str):
    min_val = min(min(preds), min(y_test))
    max_val = max(max(preds), max(y_test))

    r_value, p_value = pearsonr(preds, y_test)

    plt.plot(preds, y_test, 'o', ms=3)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # red dashed y=x line
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("Predicted PPGR")
    plt.ylabel("Measured PPGR")

    plt.title(f"{model_name} R={r_value:.3f} p-value={p_value:.3f}")

    plt.show()


def create_model_summary(
        x: list,
        y: list,
        feature_names: list,
        title: str = "",
        out: Union[str, None] = None,
        model_type: ModelType = ModelType.XGBOOST,
        n_estimators=1_000,
        learning_rate=0.05,
        max_depth=8,
        r_iterations=R_ITERATIONS,
        shap_out: Union[str, None] = None,
        pdp_out: Union[str, None] = None,
        ):
    """
    Create a .md model summary of results after r_iterations (usually 100)
    """
    print('Starting Model Summary, this may take some time...')

    output: str = ""
    if len(title) > 0:
        output += f"# {title} #"
    output += f"\n\n**Model: {model_type.name}** | n_estimators: {n_estimators}, learning_rate: {learning_rate}, max_depth: {max_depth}"
    output += "\n\n**Features:** " + ", ".join([name for name in feature_names])

    rs = []
    for i in range(r_iterations):
        preds, y_test = get_new_preds(x, y)
        r, p = pearsonr(preds, y_test)
        rs.append(r)
        if i % np.ceil(r_iterations / 10) == 0:
            print(f"r_iteration: {i}")
    output += f"\n\n**R ({r_iterations} iterations):** Mean {np.mean(rs):.3f}, Std {np.std(rs):.3f}"

    # Create model for shap/pdp analysis
    x_train, y_train, x_test, y_test = split_train_test(x, y)
    if model_type == ModelType.XGBOOST:
        model = xgboost(x_train, y_train)
    elif model_type == ModelType.GRADIENT_BOOSTING:
        model = gradient_boosting(x_train, y_train)
    else:
        raise ValueError("Model Type not supported")

    print("Starting SHAP Analysis")
    output += "\n## SHAP Analysis ##"
    SHAP_analysis(x_test, model, feature_names, shap_out)
    output += "\n\n" + (f"SHAP Analysis Plot in: {shap_out}" if shap_out is not None else\
        "SHAP Analysis Plot has not been saved.")
    output += "\n\nConsider using shap.dependence_plot for individual feature analysis"

    print("Starting PDP Analysis")
    output += "\n## PDP Analysis ##"
    PDP_analysis(model, x_test, feature_names, pdp_out)
    output += "\n\n" + (f"PDP Analysis Plot in: {pdp_out}" if shap_out is not None else\
        "PDP Analysis Plot has not been saved.")
    output += "\n\nFor categorical features consider running a PDP plot with categorical_features parameter."

    if out is None:
        print('_' * 50 + "\n" + output + '\n' + '_' * 50)
    else:
        # Save output to out file
        with open(out, "w") as f:
            f.write(output)


if __name__ == "__main__":
    x = np.load("data/CGMacros/feature_label/x.npy", allow_pickle=True)
    y = np.load("data/CGMacros/feature_label/y.npy", allow_pickle=True)
    feature_names = np.load("data/CGMacros/feature_label/feature_names.npy", allow_pickle=True)

    title = "macro_and_fiber_only"
    create_model_summary(x,
                         y,
                         feature_names,
                         out=f"results/CGMacros/model_summaries/{title}.md",
                         shap_out=f"results/CGMacros/SHAP_PDP/{title}_shap.png",
                         pdp_out=f"results/CGMacros/SHAP_PDP/{title}_pdp.png",
                         )
