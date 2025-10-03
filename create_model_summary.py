from enum import Enum
from typing import Union
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from models import get_new_preds, ModelType, SHAP_analysis, split_train_test, xgboost, gradient_boosting, PDP_analysis, \
    actual_expected_plt
from process_data.main import load_dataframe, FeatureLabelReducer

R_ITERATIONS = 100

CGMacro_USER_GROUPS = {
    "healthy": [1,  2, 4, 6, 15, 17, 18, 19, 21, 27, 31, 32, 33, 34, 48],
    "prediabetes": [7, 8,  9, 10, 11, 13, 16, 20, 22, 23, 26, 29, 41, 43, 44, 45],
    "t2dm": [3, 5, 12, 14, 28, 30, 35, 36, 38, 39, 42, 46, 47, 49]
}



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


def run_r_iterations(x: list, y: list, r_iterations: int, print_updates=False):
    rs = []
    for i in range(r_iterations):
        preds, y_test = get_new_preds(x, y)
        r, p = pearsonr(preds, y_test)
        rs.append(r)
        if print_updates and i % np.ceil(r_iterations / 10) == 0:
            print(f"r_iteration: {i}")
    return rs


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
        actual_expected_out: Union[str, None] = None
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

    rs = run_r_iterations(x, y, r_iterations, print_updates=True)
    output += f"\n\n**R ({r_iterations} iterations):** Mean {np.mean(rs):.3f}, Std {np.std(rs):.3f}"

    # Create model for shap/pdp analysis
    x_train, y_train, x_test, y_test = split_train_test(x, y)
    print(f"X train: {len(x_train)}, X_test: {len(x_test)}")
    if model_type == ModelType.XGBOOST:
        model = xgboost(x_train, y_train)
    elif model_type == ModelType.GRADIENT_BOOSTING:
        model = gradient_boosting(x_train, y_train)
    else:
        raise ValueError("Model Type not supported")

    # Predicted v Actual Graph
    # preds = model.predict(x_test)
    # plt_model_results(preds, y_test, "")

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

    print("Starting actual expected plt")
    # actual_expected_plt(preds, y_test, actual_expected_out)

    if out is None:
        print('_' * 50 + "\n" + output + '\n' + '_' * 50)
    else:
        # Save output to out file
        with open(out, "w") as f:
            f.write(output)
            print(output)


if __name__ == "__main__":
    base_file_path = "data/CGMacros/pickle/"
    df_dict = dict()
    for pkl in ["cgm", "dynamic_user", "log", "static_user"]:
        df_dict[pkl] = load_dataframe(base_file_path + pkl + ".pkl")
    feature_groups = {
        "static_user": ["Sex", "Body weight", "Height", "Self-identity"],
        "static_user2": ["UserID"],
        "log_macros": ["Energy", "Carbohydrate", "Protein", "Fat"],
        # "log_food": ["Food Types"],
        "temporal_cgm": ["cgm_p30", "cgm_p60", "cgm_p120"],
        "temporal_food": ["meal_hour", "time_since_last_meal"]
    }
    for key in ["healthy", "prediabetes", "t2dm"]:
        users = CGMacro_USER_GROUPS[key]
        new_df_dict = df_dict.copy()
        new_df_dict["cgm"] = new_df_dict["cgm"][new_df_dict["cgm"]["UserID"].isin(users)]
        new_df_dict["log"] = new_df_dict["log"][new_df_dict["log"]["UserID"].isin(users)]
        new_df_dict["dynamic_user"] = new_df_dict["dynamic_user"][new_df_dict["dynamic_user"]["UserID"].isin(users)]
        new_df_dict["static_user"] = new_df_dict["static_user"][new_df_dict["static_user"]["UserID"].isin(users)]

        reducer = FeatureLabelReducer(new_df_dict, feature_groups)
        feature_names, x, y = reducer.get_x_y_data()
        title = f"SuUiFmTgTf_{key}"
        create_model_summary(x,
                             y,
                             feature_names,
                             out=f"results/CGMacros/model_summaries/{title}.md",
                             # shap_out=f"results/CGMacros/SHAP_PDP/{title}_shap.png",
                             # pdp_out=f"results/CGMacros/SHAP_PDP/{title}_pdp.png",
                             # actual_expected_out=f"results/CGMacros/SHAP_PDP/{title}_scatter.png",
                             )
    # create_model_summary(x, y, feature_names, r_iterations=1)
