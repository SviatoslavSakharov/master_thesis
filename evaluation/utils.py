import pandas as pd
import numpy as np
import ast
import os
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol


paper_results = {
    "QED": 0.556,
    "SA": 0.729,
    "Lipinski": 4.742,
    "Vina": 7.117,
}

sns.set_style("whitegrid")  # Options include "whitegrid", "darkgrid", "white", "dark", and "ticks"

# Step 3: Configure Font and Sizes
font = {
    "family": "serif",  # Use a serif font (more suitable for scientific papers)
    "weight": "bold",  # Normal weight
    "size": 12,  # General font size, can be adjusted as needed
}
plt.rc("font", **font)

# Update specific font sizes
plt.rc("axes", titlesize=14)  # Title font size
plt.rc("axes", labelsize=12)  # X and Y label font size
plt.rc("xtick", labelsize=10)  # X-axis tick label font size
plt.rc("ytick", labelsize=10)  # Y-axis tick label font size
plt.rc("legend", fontsize=12)  # Legend font size
plt.rc("figure", titlesize=14)  # Figure title font size

# Set bold for titles and labels specifically
plt.rc("axes", titleweight="bold")  # Make axis titles bold
plt.rc("axes", labelweight="bold")  # Make axis labels bold
plt.rc("figure", titleweight="bold")  # Make the figure suptitle bold

# Step 4: Additional Plot Adjustments
plt.rc("lines", linewidth=1.5)  # Line width
plt.rc("axes", linewidth=0.75)  # Axes line width

# Optional: Set a larger figure size for better readability in papers
plt.rc("figure", figsize=(8, 6))


def read_times(folder):
    """
    Reads a txt file, separated by space, first column is the ligand name, second column is the time in seconds.

    Args:
    folder (str): folder path to the txt file.

    Returns:
    pd.DataFrame: The processed DataFrame.
    """
    # Load the CSV file
    df = pd.read_csv(folder / "pocket_times.txt", sep=" ", header=None, dtype={0: str, 1: float})
    df.columns = ["ligand", "time"]
    return df

def read_metrics(file_path):
    """
    Reads a CSV file, converts array-like strings into separate rows, and replaces "None" with NaN.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The processed DataFrame.
    """
    # Load the CSV file
    df1 = pd.read_csv(file_path / "metrics.csv")
    df1.rename(columns={"file_names": "ligand"}, inplace=True)
    if os.path.exists(file_path / "qvina" / "qvina2_scores.csv"):
        df2 = pd.read_csv(file_path / "qvina" / "qvina2_scores.csv", usecols=["ligand", "scores"])
        df2["ligand"] = df2["ligand"].apply(lambda x: x.split("/")[-1])
        df2.rename(columns={"scores": "Vina"}, inplace=True)
        df = pd.merge(df1, df2, on="ligand")
        columns_to_explode = ["QED", "SA", "Lipinski", "Vina", "indices"]
    else:
        df = df1
        columns_to_explode = ["QED", "SA", "Lipinski"]
    assert len(df) == 100, f"Expected 100 rows, but got {len(df)} rows."
    df.rename(columns={"lipinski": "Lipinski"}, inplace=True)

    # Replace "None" with NaN
    df.replace("None", np.nan, inplace=True)
    df["indices"] = [np.arange(100) for _ in range(100)]

    # Function to safely evaluate the list-like strings
    def safe_eval(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    # Apply the safe_eval function to each relevant column
    for column in columns_to_explode:
        df[column] = df[column].apply(safe_eval)

    # Explode the lists into separate rows
    df_exploded = df.explode(columns_to_explode, ignore_index=True)
    df_exploded["Vina"] = df_exploded["Vina"].apply(lambda x: np.nan if x > 0 else -x)
    assert len(df_exploded) == 10000

    return df_exploded


def print_df_info(df, name=None, metrics=["QED", "SA", "Lipinski", "Vina"]):
    if name:
        print(name)
    for metric in metrics:
        print(f"metric {metric}:")
        print(
            f"{metric} mean: {df[metric].mean():.2f}, {metric} std: {df[metric].std():.2f}, {metric} max: {df[metric].max():.2f}, {metric} min: {df[metric].min():.2f}"
        )


def plot_metrics(dfs, names, **kwargs):
    """
    Plots barplots to compare each metric from multiple dataframes using mean with std lines.
    Each metric will have its own subplot.

    Args:
    dfs (list of pd.DataFrame): The dataframes to compare.
    names (list of str): The names corresponding to each dataframe.
    """
    # Adding a source identifier
    for df, name in zip(dfs, names):
        print_df_info(df, name)

    for i, df in enumerate(dfs):
        df["Source"] = names[i]

    # Combining the DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    # List of metrics to compare (excluding the first column which might be an identifier and the last 'Source' column)
    metrics = kwargs.get("metrics", ["Vina", "QED", "SA", "Lipinski"])
    rows = kwargs.get("rows", 1)
    cols = len(metrics) // rows
    figsize = kwargs.get("figsize", (15, 5))
    suptitle = kwargs.get("suptitle", None)
    save = kwargs.get("save", None)
    include_paper_results = kwargs.get("include_paper_results", False)

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    colors = sns.color_palette("tab10", n_colors=len(names))
    # Plot each metric in a separate subplot boxplot
    for i, metric in enumerate(metrics):
        to_plot = combined_df[["Source", metric]]
        if metric == "Lipinski":
            sns.barplot(
                data=to_plot, x="Source", y=metric, ax=axes[i], errorbar="sd", capsize=0.1, palette=colors, hue="Source"
            )
        else:
            sns.violinplot(data=to_plot, x="Source", y=metric, ax=axes[i], palette=colors, hue="Source")
        if include_paper_results:
            axes[i].axhline(paper_results[metric], color="red", linestyle="--", label="Paper Results")
        y_label = metric if metric != "Vina" else "-Vina"
        title = metric + "↑" if metric != "Vina" else "-Vina ↑"
        axes[i].set_title(title)
        axes[i].set_ylabel(y_label)
        axes[i].set_xlabel(None)
        # rotate x-axis labels
        for tick in axes[i].get_xticklabels():
            tick.set_rotation(45)
    if title:
        fig.suptitle(suptitle)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
    else:
        plt.tight_layout()
    if save:
        plt.savefig(save)

    return None


def plot_2d_ligands(df, top_ligands, save=None, suptitle=None):
    # Plot QED vs SA
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs = axs.flatten()
    N = len(top_ligands)
    label = f"Top {N} Ligands"

    sns.scatterplot(data=df, x="QED", y="SA", color="red", label="Other Ligands", ax=axs[0])
    sns.scatterplot(data=top_ligands, x="QED", y="SA", color="blue", label=label, s=50, ax=axs[0])
    axs[0].set_title("QED vs SA")

    sns.scatterplot(data=df, x="SA", y="Vina", color="red", label="Other Ligands", ax=axs[1])
    sns.scatterplot(data=top_ligands, x="SA", y="Vina", color="blue", label=label, s=50, ax=axs[1])
    axs[1].set_title("SA vs -Vina")
    axs[1].set_ylabel("-Vina")

    sns.scatterplot(data=df, x="QED", y="Vina", color="red", label="Other Ligands", ax=axs[2])
    sns.scatterplot(data=top_ligands, x="QED", y="Vina", color="blue", label=label, s=50, ax=axs[2])
    axs[2].set_title("QED vs -Vina")
    axs[2].set_ylabel("-Vina")
    plt.legend()
    plt.suptitle(suptitle)
    plt.tight_layout()
    if save:
        fig.savefig("F:\\Studium\\MS\\thesis\\master_thesis\\img\\" + save)

    return fig


def find_top_N_single(df, N=5, weights=(1, 1, 2)):
    # 1. Drop all rows where at least 1 element is nan
    df = df.dropna()
    # print(f"Finding top {N} ligands...")

    def min_max_scale(series):
        return (series - series.min()) / (series.max() - series.min())

    qed = min_max_scale(df["QED"]).values
    sa = min_max_scale(df["SA"]).values
    vina = min_max_scale(df["Vina"]).values

    score = (weights[0] * qed + weights[1] * sa + weights[2] * vina) / sum(weights)
    top_N = np.argsort(score)[-N:]
    top_N_ligands = df.iloc[top_N]

    return top_N_ligands


def find_top_N(df, N=5, random_pocket=None, save=None, suptitle=None, weights=(1, 1, 2)):
    # print(f"Finding top {N} ligands...")
    ligands = df["ligand"].unique()
    random_i = random_pocket if random_pocket else np.random.randint(0, len(ligands))
    print(f"Random pocket id: {random_i}, name: {ligands[random_i]}")
    top_N_ligands = []
    for i, ligand in enumerate(ligands):
        df_ligand = df[df["ligand"] == ligand]
        top_N_ligands.append(find_top_N_single(df_ligand, N, weights=weights))
        if i == random_i:
            pass
            # plot_2d_ligands(df_ligand, top_N_ligands[-1], save=save, suptitle=suptitle)
    top_N_ligands = pd.concat(top_N_ligands)
    return top_N_ligands


def read_molecules(basedir):
    folder = basedir / "processed"
    mols = {}
    sdf_files = list(folder.glob("*.sdf"))

    for mol_file in sdf_files:
        with open(mol_file, "r") as f:
            mols_str = f.read().split("$$$$\n")[0:-1]
            mols[mol_file.stem + ".sdf"] = mols_str
    return mols


def plot_mols(mols_arr, dfs_topN, names, file_name=None, true_mols_dir=None):
    assert len(mols_arr) == len(dfs_topN) == len(names)
    files = list(mols_arr[0].keys())
    if file_name:
        random_file = file_name
    else:
        random_file = np.random.choice(files)
    true_mol_file = random_file.replace("_gen", "")
    if true_mols_dir:
        true_mol = open(true_mols_dir / true_mol_file, 'r').read()
    else:
        true_mol = None
    print(f"File: {random_file}")
    cols = 6 if true_mol else 5
    rows = len(mols_arr)
    print(f"Rows: {rows}, Cols: {cols}")
    view = py3Dmol.view(
        width=1500, height=int(rows * 250), linked=False, viewergrid=(rows, cols), js="https://3dmol.org/build/3Dmol.js"
    )
    print(len(dfs_topN))
    for row in range(rows):
        print(f"row {row}: {names[row]}")
        topN = dfs_topN[row]
        topN_sample = topN[topN["ligand"] == random_file]
        indices = topN_sample["indices"].values.astype(int)
        topN_mols = np.array(mols_arr[row][random_file])[indices]
        for i in range(cols):
            if i == 0 and true_mol:
                view.addModel(true_mol, format="sdf", viewer=(row, 0))
                view.setStyle({"model": -1}, {"stick": {}}, viewer=(row, 0))
            elif true_mol:
                view.addModelsAsFrames(topN_mols[i-1], viewer=(row, i))
            else:
                view.addModelsAsFrames(topN_mols[i], viewer=(row, i))
            view.addStyle({"model": -1}, {"stick": {}}, viewer=(row, i))

    view.zoomTo()
    return view


def plot_lineplot_vs(plot_dict, save=None, suptitle=None, **kwargs):
    """
    Plots lineplots to compare each metric from multiple dataframes using mean with std lines.
    Each metric will have its own subplot.

    Args:
    plot_dict (dict): The dataframes to compare. {"label": {"dfs": [...], "names": [...]}}
    """

    # List of metrics to compare (excluding the first column which might be an identifier and the last 'Source' column)
    metrics = kwargs.get("metrics", ["Vina", "QED", "SA", "Lipinski"])
    rows = kwargs.get("rows", 1)
    cols = len(metrics) // rows
    figsize = kwargs.get("figsize", (15, 5))
    xlabel = kwargs.get("xlabel", None)
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    # Plot each metric in a separate subplot boxplot
    for i, metric in enumerate(metrics):
        dfs = []
        for label, data in plot_dict.items():
            dfs_label = data["dfs"]
            names_label = data["names"]
            for j, df in enumerate(dfs_label):
                df["Label"] = label
                df["steps"] = int(names_label[j].split("_")[1])
            # Combining the DataFrames
            dfs.extend(dfs_label)
        combined_df = pd.concat(dfs, ignore_index=True)
        sns.lineplot(
            data=combined_df, 
            x="steps", 
            y=metric, 
            ax=axes[i], 
            hue="Label",
            errorbar=("se", 2),
            marker="o",)

        axes[i].set_title(metric)
        axes[i].set_ylabel(metric)
        if xlabel:
            axes[i].set_xlabel(xlabel)

        unique_steps = sorted(combined_df["steps"].unique())
        axes[i].set_xticks(unique_steps)
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    if save:
        plt.savefig(save)
    return None
        

def plot_times(dfs, names, **kwargs):
    for i, df in enumerate(dfs):
        df["step"] = names[i].split("_")[1]
    combined_df = pd.concat(dfs, ignore_index=True)
    figsize = kwargs.get("figsize", (15, 5))
    save = kwargs.get("save", None)
    fig, ax = plt.subplots(figsize=figsize)
    line_plot = sns.lineplot(data=combined_df, x="step", y="time", marker="o", errorbar="sd", ax=ax)
    line_data = line_plot.lines[0].get_xydata()
    steps = line_data[:, 0]
    mean_times = line_data[:, 1]
    max_time = mean_times.max()
    for step, mean_time in zip(steps, mean_times):
        speed_up = max_time / mean_time
        ax.annotate(f"{speed_up:.1f}x", (step, mean_time), textcoords="offset points", xytext=(0,10), ha='center')
    title = kwargs.get("title", "Sampling Time vs Number of Steps")
    plt.title(title)
    plt.xlabel("Number of Steps")
    plt.ylabel("Sampling Time in Seconds")
    plt.tight_layout()
    if save:
        plt.savefig(save)
    return None
