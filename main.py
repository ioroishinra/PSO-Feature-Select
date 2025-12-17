import math
import yaml
import random
import logging.config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

logger = logging.getLogger("PSO")

def k_fold_split(x: pd.DataFrame, y: pd.Series, k: int, shuffle=True, random_seed=None):
    n = len(x)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)

    fold_sizes = np.full(k, n // k)
    fold_sizes[:n % k] += 1

    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_val = x.iloc[val_idx]
        y_val = y.iloc[val_idx]

        yield x_train, y_train, x_val, y_val
        current = stop

def grid_shape(n):
    root = int(math.sqrt(n))
    for r in range(root, 0, -1):
        if n % r == 0:
            return r, n // r
    return 1, n

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

    # PSO optimization
    for _ in trange(info["num_iterations"], desc="PSO optimization: ", unit="Iters", leave=False):

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
        best_scores.append(global_best_score.values[0][0])

        # x generation
    x_tr_pso = x.iloc[:, (global_best == 1).values]

    return x_tr_pso, global_best, best_scores

def main(info):
    # random seed setting
    random.seed(info["random_seed"])
    np.random.seed(info["random_seed"])

    # dataset creation
    data = pd.read_csv(info["dataset_file_path"])
    x = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    logger.info(f"Dataset loaded with shape: {data.shape}")

    # dataset split
    cv_scores = []
    cv_scores_pso = []
    all_best_scores = []
    for fold, (x_tr, y_tr, x_val, y_val) in enumerate(k_fold_split(x, y, info["k_fold"], shuffle=info["shuffle"], random_seed=info["random_seed"])):

    # feature engineering
        x_tr_pso, global_best, best_scores = pso(x_tr, y_tr, info)
        all_best_scores.append(best_scores)
        x_val_pso = x_val.iloc[:, (global_best == 1).values]
        result = ""
        for feature in global_best.values:
            result += str(feature)
        logger.info(result)
        logger.info(f"Fold {fold+1} Global best: {sum(global_best.values)} features, {result}")

    # model creation
        model = model_generator(info["model_type"])
        model_pso = model_generator(info["model_type"])

    # model training
        model.fit(x_tr, y_tr)
        model_pso.fit(x_tr_pso, y_tr)

    # model evaluation
        score = np.mean((y_val - model.predict(x_val)) ** 2)
        cv_scores.append(score)
        logger.info(f"ORI Score: {score}")
        score_pso = np.mean((y_val - model_pso.predict(x_val_pso)) ** 2)
        cv_scores_pso.append(score_pso)
        logger.info(f"PSO Score: {score_pso}")
    logger.info(f"ORI Mean Score: {np.mean(cv_scores)}")
    logger.info(f"PSO Mean Score: {np.mean(cv_scores_pso)}")

    # visualization
    plt.figure(figsize=(8, 5))
    for i, scores in enumerate(all_best_scores):
        plt.plot(range(1, len(scores) + 1), scores, label=f"Fold {i + 1}")
    plt.xlabel("Iteration")
    plt.ylabel("Best MSE")
    plt.title("PSO Global Best Score over Iterations (k-fold CV)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # logger configuration
    with open("config/logging.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["handlers"]["file"]["filename"] = f"logs/exp_{timestamp}.log"
    logging.config.dictConfig(config)

    # parameters loading
    with open("config/param.yaml", "r", encoding="utf-8") as f:
        info = yaml.safe_load(f)
    logger.info("=== Experiment parameters ===")
    for key, value in info.items():
        logger.info(f"{key:<20} | {value}")

    main(info)