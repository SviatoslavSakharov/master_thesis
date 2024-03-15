import numpy as np
import pandas as pd


def read_line(line):
    score, values = line.split(":")
    values = values.strip().split("+-")
    return score, float(values[0]), float(values[1])

def read_scores(prediction_path, qvina=False, top10=False):
    if qvina:
        path = prediction_path / "evaluation_qvina.txt"
    else:
        if top10:
            path = prediction_path / "evaluation_top10.txt"
        else:
            path = prediction_path / "evaluation.txt"
    with open(path, "r") as f:
        lines = f.readlines()
        if not qvina:
            lines = lines[1:] # skip header
    scores, means, stds = [], [], []
    for line in lines:
        score, mean, std = read_line(line)
        if score == "logP":
            continue
        scores.append(score)
        means.append(mean)
        stds.append(std)
    return scores, means, stds

def normalize_btw_a_b(arr, a, b):
    return (b-a) * (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) + a