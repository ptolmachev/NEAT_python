# Not currently working example of how to do a hyperparameter search

import optuna
from functools import partial
import numpy as np
from src.evolution.evolving.evolve_CartPole import run_evolution

np.set_printoptions(suppress=True)
import warnings
warnings.simplefilter("ignore", UserWarning)
import json
import sys
import os
from copy import deepcopy
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "../"))
sys.path.insert(0, os.path.join(current_dir, "../../"))
sys.path.insert(0, os.path.join(current_dir, "../../../"))
from src.evolution.InnovationHandler import InnovationHandler
from src.evolution.Nature_old import Nature
from src.evolution.Animal import Animol
from src.evolution.Blueprint import BluePrint
import numpy as np
import gymnasium

def objective(trial, num_repeats=7):
    env_name = 'CartPole-v1'
    n_inputs = 3
    n_outputs = 2
    n_generations = 200
    max_animols = 100
    save_every = 50
    render = False
    render_every = -1
    num_eval_repetitions = 11
    action_noise = 0.03
    neuron_type = 'relu'
    weight_init_std = 0.2
    weight_change_std = 0.05
    data_folder = '../../data/evolved_models/CartPole-v1'
    orph_node_thr = 0.05

    # define params to be varied
    c_d = trial.suggest_float('c_d', 0.1, 1.0)
    c_w = trial.suggest_float('c_w', 0.1, 1.0)
    mu = trial.suggest_float('mu', 0.05, 2.0)
    rel_fitness_advantage = trial.suggest_float('rel_fitness_advantage', 0.01, 0.5, log=True)

    best_scores = []
    for i in range(num_repeats):
        env = gymnasium.make(env_name)
        best_score = run_evolution(env=env,
                       n_inputs=n_inputs,
                       n_outputs=n_outputs,
                       n_generations=n_generations,
                       max_animols=max_animols,
                       rel_fitness_advantage=rel_fitness_advantage,
                       c_d=c_d,
                       c_w=c_w,
                       mu=mu,
                       num_eval_repetitions=num_eval_repetitions,
                       action_noise=action_noise,
                       neuron_type=neuron_type,
                       weight_init_std=weight_init_std,
                       weight_change_std=weight_change_std,
                       orph_node_thr=orph_node_thr,
                       save_every=save_every,
                       render=render,
                       render_every=render_every, data_folder=data_folder)

        print(f"best_score in this run {best_score}")
        best_scores.append(deepcopy(best_score))
    print(f"best_score averaged across trials {np.mean(best_scores)}")
    return np.mean(best_scores)

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    num_repeats = 5
    n_trials = 10
    objective = partial(objective, num_repeats=num_repeats)
    study.optimize(objective, n_trials=n_trials)
    print(f"best parameters : {study.best_params}")
