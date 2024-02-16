from datetime import datetime
from src.evolution.Nature import Nature
from src.evolution.Logger import Logger
from src.evolution.InnovationHandler import InnovationHandler
import numpy as np
import gymnasium
import hydra
import sys
from src.slimevolleygym.slimevolley import SlimeVolleyEnv
import ast


class Validator():
    def __init__(self, env_builder_function, eval_repeats, max_timesteps):
        self.environment = env_builder_function()
        self.eval_repeats = eval_repeats
        self.max_timesteps = max_timesteps

    def prepare_environment(self, seed):
        try:
            self.environment.seed(seed=seed)
            obs = self.environment.reset()
        except:
            obs = self.environment.reset(seed=seed)
        if type(obs) == tuple:
            obs = obs[0]
        return obs

    def run_through_environment(self, animal, seed):
        animal.action_noise = 0 #its a serious test. Focus!
        obs = self.prepare_environment(seed)
        total_reward = 0
        done = False
        while_cnt = 0
        while (not done) and (while_cnt < self.max_timesteps):
            action = animal.react(inputs=obs)
            result = self.environment.step(action=action)
            if len(result) == 4: obs, reward, done, info = result
            elif len(result) == 5: obs, reward, done, _, info = result
            total_reward += reward
            while_cnt += 1
        return total_reward

    def get_validation_score(self, animal):
        rewards = []
        for i in range(self.eval_repeats):
            reward = self.run_through_environment(animal, seed=np.random.randint(100000))
            rewards.append(reward)
        self.environment.close()
        return np.nanmean(rewards)

env_name = "Pendulum-v1"
@hydra.main(config_path="conf", config_name=f"config_{env_name}", version_base="1.3")
def run_evolution(cfg):
    for i in range(1):
        innovation_handler = InnovationHandler(cfg.innovation_handler_params["maxlen"])
        innovation_handler.innovation_counter = 0

        env_name = cfg.env_name
        if env_name == "SlimeVolley-v1":
            environment_builder_fn = SlimeVolleyEnv
        else:
            try:
                environment_builder_fn = lambda: gymnasium.make(env_name)
            except:
                raise ValueError(f"The environment {env_name} doesn't exist in gymnasium")

        print(f"Evolving agents to solve {env_name}.")
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%H%M_%d_%m_%Y")

        logger = Logger(log_folder=cfg.logger_params.log_folder,
                        img_folder=cfg.logger_params.img_folder,
                        data_folder=cfg.logger_params.data_folder,
                        render=cfg.logger_params.render,
                        render_every=cfg.logger_params.render_every,
                        plot_every=cfg.logger_params.plot_every,
                        tag=formatted_datetime)


        animal_params = dict(cfg.animal_params)
        if not(animal_params["action_bounds"] is None):
            animal_params["action_bounds"] = ast.literal_eval(animal_params["action_bounds"])
        #define Nature
        nature = Nature(innovation_handler=innovation_handler,
                        environment_builder_fn=environment_builder_fn,
                        max_animals=cfg.nature_params.max_animals,
                        max_species=cfg.nature_params.max_species,
                        max_timesteps=cfg.nature_params.max_timesteps,
                        animal_params=animal_params,
                        blueprint_params = dict(cfg.blueprint_params),
                        rel_advantage_ind=cfg.nature_params.rel_advantage_ind,
                        rel_advantage_spec=cfg.nature_params.rel_advantage_spec,
                        mutation_probs=ast.literal_eval(cfg.nature_params.mutation_probs),
                        syn_mutation_prob=cfg.nature_params.syn_mutation_prob,
                        syn_mut_type_probs=ast.literal_eval(cfg.nature_params.syn_mut_type_probs),
                        weight_change_std=cfg.nature_params.weight_change_std,
                        c_d=cfg.nature_params.c_d,
                        c_w=cfg.nature_params.c_w,
                        delta=cfg.nature_params.delta,
                        gamma=cfg.nature_params.gamma,
                        cull_ratio=cfg.nature_params.cull_ratio,
                        parthenogenesis_rate=cfg.nature_params.parthenogenesis_rate,
                        interspecies_mating_chance=cfg.nature_params.interspecies_mating_chance,
                        metabolic_penalty=cfg.nature_params.metabolic_penalty,
                        eval_repeats=cfg.nature_params.eval_repeats,
                        parallel=cfg.nature_params.parallel,
                        num_workers=cfg.nature_params.num_workers,
                        self_play=cfg.nature_params.self_play,
                        num_reference_animals=cfg.nature_params.num_reference_animals,
                        logger=logger,
                        validator=Validator(eval_repeats=13, max_timesteps=1000, env_builder_function=environment_builder_fn))
        nature.evolve_loop(n_generations=cfg.evolution_params.n_generations)
        del innovation_handler
    return None

if __name__ == '__main__':
    run_evolution()