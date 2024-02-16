import os
import json
import time

import numpy as np
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
from src.evolution.Blueprint import get_neurons_by_type#, get_adjacency_matrix
from src.evolution.utils import jsonify


class Logger():
    def __init__(self, log_folder, data_folder, img_folder, tag, render=False, render_every=100, plot_every=10):
        self.log_folder = log_folder
        self.data_folder = data_folder
        self.img_folder = img_folder
        if not log_folder is None:
            try:
                os.makedirs(log_folder, exist_ok=True)
            except:
                pass
        if not data_folder is None:
            try:
                os.makedirs(data_folder, exist_ok=True)
            except:
                pass
        if not img_folder is None:
            try:
                os.makedirs(img_folder, exist_ok=True)
            except:
                pass
        self.tag = tag
        self.render = render
        self.render_every = render_every
        self.plot_every = plot_every

    def save_log(self, log_dict, file_name):
        file_path = os.path.join(self.log_folder, file_name)
        json.dump(log_dict, open(file_path, "w", encoding="utf8"), indent=4)
        return None

    def save_data(self, data_dict, file_name):
        file_path = os.path.join(self.data_folder, file_name)
        json.dump(data_dict, open(file_path, "w", encoding="utf8"), indent=4)
        return None

    def fossilize(self, top_animol, generation, env_name, score=None):
        fittest_animol_genome = top_animol.blueprint.genome_dict
        data_dict = {}
        n_hidden_nrns = int(len(get_neurons_by_type(fittest_animol_genome, "h")))
        # n_synapses = int(np.sum(get_adjacency_matrix(fittest_animol_genome)))
        # data_dict["N synapses"] = n_synapses
        data_dict["N hidden nrns"] = n_hidden_nrns
        data_dict["genome dict"] = fittest_animol_genome

        if score is None:
            score = np.round(top_animol.fitness, 3)
        file_name = f"{env_name}_generation={generation + 1}_score={score}_N={n_hidden_nrns}.json"
        try:
            self.save_data(data_dict, file_name)
        except:
            self.save_data(data_dict, file_name)
        return None

    def plot_MDS_embedding(self, D, file_name):
        # MDS embedding of the animol genomes
        mds = MDS()
        mds.fit_transform(D)
        fig = plt.figure(figsize = (5, 5))
        plt.scatter(mds.embedding_[:, 0], mds.embedding_[:, 1], color = 'r', edgecolors='k')
        fig.savefig(os.path.join(self.img_folder, file_name))
        plt.close()

    def plot_scores(self, top_scores, mean_scores, std_scores,  file_name, val_scores = None):
        fig = plt.figure(figsize = (10, 4))
        top_scores = np.array(top_scores)
        mean = np.array(mean_scores)
        std = np.array(std_scores)
        if not(val_scores is None):
            plt.plot(val_scores, color = 'g', label="validation scores")

        plt.plot(top_scores, color = 'r', label="top scores")
        plt.plot(mean, color='blue', label="mean scores")
        plt.plot(mean - std, color='blue', linestyle='--', linewidth = 0.5)
        plt.plot(mean + std, color='blue', linestyle='--', linewidth = 0.5)

        # getting a ylim
        if not(val_scores is None):
            ref_scores = val_scores
        else:
            ref_scores = top_scores
        y_min = np.min(ref_scores)
        y_max = np.max(ref_scores)
        range = y_max - y_min
        y_mean = (y_min + y_max) / 2
        z_min = y_mean - 1.1 * range / 2
        z_max = y_mean + 1.1 * range / 2
        plt.ylim([z_min, z_max])

        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.1, label='±1 Std Dev')
        plt.grid(True)
        plt.xlabel("Generation")
        plt.ylabel("Achieved score")
        plt.legend()
        fig.savefig(os.path.join(self.img_folder, file_name))
        plt.close()
        return None

    def plot_top_circuit(self):
        pass
        return None

    def run_demonstration(self, animal, environment, max_timesteps= 1000, sleep=0.0005):
        seed = np.random.randint(100000)

        try:
            obs = environment.reset(seed=seed)
        except:
            obs = environment.reset()
            environment.seed(seed=seed)

        environment.render()
        if type(obs) == tuple:  # for compatibility with other gym environments
            obs = obs[0]
        total_reward = 0

        for i in range(max_timesteps):
            environment.render()
            time.sleep(sleep)
            action = animal.react(inputs=obs)
            result = environment.step(action=action)
            match len(result):
                case 3: obs, reward, done = result
                case 4: obs, reward, done, info = result
                case 5:  obs, reward, done, _, info = result
            total_reward += reward
            if done:
                break
        environment.close()
        return total_reward

