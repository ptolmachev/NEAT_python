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

        #define Nature
        animal_params = cfg.animal_params
        animal_params["action_bounds"] = ast.literal_eval(animal_params["action_bounds"])
        nature = Nature(innovation_handler=innovation_handler,
                        environment_builder_fn=environment_builder_fn,
                        max_animals=cfg.nature_params.max_animals,
                        max_species=cfg.nature_params.max_species,
                        max_timesteps=cfg.nature_params.max_timesteps,
                        animal_params=animal_params,
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
                        logger=logger)
        nature.evolve_loop(n_generations=cfg.evolution_params.n_generations)
        del innovation_handler
    return None

if __name__ == '__main__':
    run_evolution()