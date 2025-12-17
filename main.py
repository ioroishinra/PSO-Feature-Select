import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from utils import *

def model_generator(model_type):
    if model_type == "lr":
        return LinearRegression()
    if model_type == "svr":
        return SVR(kernel="rbf")
    else:
        raise NotImplementedError

def function_generator(eval_type, model_type):
    def predict(x, y):
        model = model_generator(model_type)
        model.fit(x, y)
        return model.predict(x)
    if eval_type == "mse":
        return lambda x, y: np.mean((y - predict(x, y)) ** 2)
    else:
        raise NotImplementedError

def get_neighbourhood(topology_type, particles, idx, k):
    if topology_type == "von_neumann":
        rows, cols = grid_shape(particles.shape[0])
        r = idx // cols
        c = idx % cols
        up = ((r - 1) % rows) * cols + c
        down = ((r + 1) % rows) * cols + c
        left = r * cols + ((c - 1) % cols)
        right = r * cols + ((c + 1) % cols)
        return [up, down, left, right]
    elif topology_type == "global":
        return [i for i in range(particles.shape[0]) if i != idx]
    elif topology_type == "ring":
        return [(idx + i) % particles.shape[0] for i in range(-k, k + 1) if i != 0]
    elif topology_type == "random":
        k = max(1, particles.shape[0] // 5)
        return list(np.random.choice([i for i in range(particles.shape[0]) if i != idx], size=k, replace=False))
    else:
        raise NotImplementedError

def pso(x, y, info):
    print(f"## PSO optimization initializing. ")
    best_scores = []

    # fitness function definition
    func = function_generator(info["eval_type"], info["eval_model_type"])

    # initialization
    particles = pd.DataFrame(np.random.randint(0, 2, (info["num_particles"], info["dim"])))
    scores = pd.DataFrame(np.zeros((info["num_particles"])))
    for i in range(info["num_particles"]):
        scores.iloc[i] = func(x.iloc[:, (particles.iloc[i] == 1).values], y)
    velocities = pd.DataFrame(np.random.uniform(-1, 1, (info["num_particles"], info["dim"])))
    particles_best = particles.copy()
    scores_best = scores.copy()
    global_best_idx = scores_best.idxmax()
    global_best = particles_best.iloc[global_best_idx].copy()
    global_best_score = scores_best.iloc[global_best_idx].copy()
    print(f"## PSO optimization initialized. ")

    # PSO optimization
    print(f"## PSO optimization started. ")
    for iteration in range(info["num_iterations"]):

        # position update
        particles = (particles + velocities).clip(0,1).round().astype(int)

        # fitness scores update
        for i in range(info["num_particles"]):
            scores.iloc[i] = func(x.iloc[:, (particles.iloc[i] == 1).values], y)

        # position best update
        particles_best = pd.DataFrame(
            np.where(scores_best < scores, particles_best, particles),
            columns=particles.columns
        )

        # neighbor best selection
        for i in range(info["num_particles"]):
            neighbors_idx = get_neighbourhood(info["topology_type"], particles, i, info["k"])
            neighbor_best_idx = scores.loc[neighbors_idx].idxmin()
            neighbor_best = particles.iloc[neighbor_best_idx]

        # velocity update
            velocities.iloc[i] = (
                info["w"] * velocities.iloc[i] +
                info["c1"] * random.uniform(0,1) * (particles_best.iloc[i] - particles.iloc[i]) +
                info["c2"] * random.uniform(0,1) * (neighbor_best - particles.iloc[i])
            )

        # best solution search
            current_best_idx = scores.idxmin()
            current_best_score = scores.iloc[current_best_idx]
            if current_best_score.values < global_best_score.values:
                global_best_score = current_best_score.copy()
                global_best = particles.iloc[int(current_best_idx.iloc[0])]
        print(f"## Iteration {iteration+1}: {global_best_score.values[0][0]}")
        best_scores.append(global_best_score.values[0][0])

        # x generation
    x_pso = x.iloc[:, (global_best == 1).values]

    return x_pso, global_best, best_scores

def main(info):
    # dataset creation
    data = pd.read_csv(info["dataset_file_path"])
    x = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    print(f"## Dataset loaded. ")

    # feature engineering
    x_pso, global_best, best_scores = pso(x, y, info)
    print(f"## Global best: {global_best.values}")

    # model creation
    model = model_generator(info["model_type"])
    model_pso = model_generator(info["model_type"])

    # model training
    model.fit(x, y)
    model_pso.fit(x_pso, y)

    # model evaluation
    score = model.score(x, y)
    score_pso = model_pso.score(x_pso, y)
    print(f"## Score: {score}")
    print(f"## Score pso: {score_pso}")

    # visualization
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(best_scores) + 1), best_scores, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Best MSE")
    plt.title("PSO Global Best Score over Iterations")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    info = {
        # model
        "model_type": "lr",

        # datasets
        "dataset_name": None,
        "dataset_file_path": "data.csv",
        "dim": 50,

        # algorithm
        "eval_type": "mse",
        "eval_model_type": "lr",
        "num_particles": 50,
        "num_iterations": 70,
        "topology_type": "random",
        "k": 5,
        "c1": 1.0,
        "c2": 2.0,
        "w": 1.0
    }
    main(info)