import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import dotenv
import ray
from omegaconf import OmegaConf
from ray import tune   # uses ray 1.13. ctrl+c stops the training
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from utils import make_data_tag, trial_function

# ------------- Distributed training settings ------------------
DISTRIBUTED_TRAINING = False   # If True creates an instance of TorchTrainer

if DISTRIBUTED_TRAINING:
    from ctd.task_modeling.task_train_prep_parallel import train_parallel
else:
    from ctd.task_modeling.task_train_prep import train

# TODO: write down an easier logic to figure out the cpus, gpus...

# --------------------------------------------------------------------

# Add custom resolver to create the data_tag so it can be used for run dir
OmegaConf.register_new_resolver("make_data_tag", make_data_tag)
log = logging.getLogger(__name__)

# TODO: major change - from the tune.run method to the Tuner API, to support distributed training
# TODO: integrate the tas_train_prep with the tune object creation here (implement lightning and ray in one file)
# leave the highest level user interface only to choose algs, schedulers, params, and search configs

# ---------------Options---------------
LOCAL_MODE = False  # Set to True to run locally (for debugging) but also to distribute across clusters
OVERWRITE = True  # Set to True to overwrite existing run
WANDB_LOGGING = False  # Set to True to log to WandB (need an account)

RUN_DESC = "node_CDM_test_mini"  # For WandB and run dir
TASK = "MultiTask"  # Task to train on (see configs/task_env for options)
MODEL = "NODE"  # Model to train (see configs/model for options)

# -----------------Parameter Selection ---------------------------------
SEARCH_SPACE = dict(
    model = dict(
        # not much improvement of 10 over 3
        latent_size = tune.grid_search([3, 5]),
        # gating_linear = tune.grid_search([True]), 
        # gating_n_layers = tune.grid_search([1,10])    ,  
        # euler_step_size = tune.grid_search([0.1, 0.01]),
    ),
    task_wrapper=dict(
        # Task Wrapper Parameters - high learning rate, low weight decay
        weight_decay=tune.grid_search([1e-9]),
        learning_rate=tune.grid_search([5e-2]),
    ),
    trainer=dict(
        # Trainer Parameters 
        max_epochs=tune.choice([10]),
        log_every_n_steps=tune.choice([1]),
    ),
    # Data Parameters 
    params=dict(
        seed=tune.grid_search([0]),
        batch_size=tune.choice([4]),  
        #num_workers=tune.choice([1, 10]),   # TODO: default 4. these are not the same as ray workers. for dataloading. usually set = CPUs
        n_samples=tune.choice([10]),      # TODO: more? NOT the same as samples from ray tune
    ),
    # task parameters
    env_task=dict(
        task_list=tune.choice([["ContextIntMod1"]]),
    ),
    # simulation task params
    env_sim=dict(
        task_list=tune.choice([["ContextIntMod1"]]),
    ),
)


# ------------------Data Management Variables --------------------------------

dotenv.load_dotenv()
HOME_DIR = Path(os.environ.get("HOME_DIR"))

path_dict = dict(
    tt_datasets=HOME_DIR / "content" / "datasets" / "tt",
    sim_datasets=HOME_DIR / "content" / "datasets" / "sim",
    dt_datasets=HOME_DIR / "content" / "datasets" / "dt",
    trained_models=HOME_DIR / "content" / "trained_models",
)
# Make the directories if they don't exist
for key, val in path_dict.items():
    if not val.exists():
        val.mkdir(parents=True)

DATE_STR = datetime.now().strftime("%Y%m%d")
RUN_TAG = f"{DATE_STR}_{RUN_DESC}"
RUN_DIR = HOME_DIR / "content" / "runs" / "task-trained" / RUN_TAG

# -----------------Default Parameter Sets -----------------------------------
config_dict = dict(
    task_wrapper=Path(f"configs/task_wrapper/{TASK}.yaml"),
    env_task=Path(f"configs/env_task/{TASK}.yaml"),
    env_sim=Path(f"configs/env_sim/{TASK}.yaml"),
    datamodule_task=Path(f"configs/datamodule_train/datamodule_{TASK}.yaml"),
    datamodule_sim=Path(f"configs/datamodule_sim/datamodule_{TASK}.yaml"),
    model=Path(f"configs/model/{MODEL}.yaml"),
    simulator=Path(f"configs/simulator/default_{TASK}.yaml"),
    callbacks=Path(f"configs/callbacks/default_{TASK}.yaml"),
    loggers=Path("configs/logger/default.yaml"),
    trainer=Path("configs/trainer/default.yaml"),
)

if not WANDB_LOGGING:
    config_dict["loggers"] = Path("configs/logger/default_no_wandb.yaml")
    config_dict["callbacks"] = Path("configs/callbacks/default_no_wandb.yaml")

# ------------- Settings for parallel training -------------------------

# scaling_config = ScalingConfig(    
#     num_workers=4,  # number of distributed workers - should match total trials?
#     use_gpu=True,    # whether each worker should use a gpu
#     resources_per_worker={"CPU": 1, "GPU": 1},   # default, but consider cluster architecture to change
#     placement_strategy="SPREAD",   # PACK favors memory (locality) and SPREAD favors speed
# )

# run_config = RunConfig(           
#     # checkpoint_config=CheckpointConfig(
#     #     num_to_keep=3,  
#     #     checkpoint_score_attribute="loss",
#     #     checkpoint_score_order="max",
#     # ),
#     storage_path=str(RUN_DIR),
#     verbose=1,
#     progress_reporter=CLIReporter(
#             metric_columns=["loss", "training_iteration"],
#             sort_by_metric=True,
#         ),
# )

# # THIS is the highest level training object ran on each worker
# ray_trainer = TorchTrainer(
#     train,     
#     scaling_config=scaling_config,
#     run_config=run_config,
# )

# -------------------Main Function----------------------------------
def main(
    run_tag_in: str,
    path_dict: str,
    config_dict: dict,
):
    if LOCAL_MODE:
        ray.init(local_mode=True)
    if RUN_DIR.exists() and OVERWRITE:
        shutil.rmtree(RUN_DIR)

    RUN_DIR.mkdir(parents=True)
    shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)

    # tune.run(
    #     tune.with_parameters(train,run_tag=run_tag_in,path_dict=path_dict,config_dict=config_dict),   # DONE - torchtrainer
    #     metric="loss",   # DONE - in tuner object
    #     mode="min",     # DONE - in tuner object
    #     config=SEARCH_SPACE,  # DONE - in tuner object
    #     resources_per_trial=dict(cpu=1, gpu=1),    # DONE - runconfig
    #     num_samples=1,                  # DONE - in tuner object
    #     storage_path=str(RUN_DIR),   #DONE - runconfig
    #     search_alg=BasicVariantGenerator(),
    #     scheduler=FIFOScheduler(),
    #     verbose=1,   #DONE - runconfig
    #     progress_reporter=CLIReporter(    #DONE - runconfig
    #         metric_columns=["loss", "training_iteration"],
    #         sort_by_metric=True,
    #     ),
    #     trial_dirname_creator=trial_function,
    # )
    
    if DISTRIBUTED_TRAINING:
        
        # all arguments in scaling_config and run_config are untunable
        scaling_config = ScalingConfig(    
                                       # maybe 1 worker and just tuning cpu/gpu OR change placement_strategy
                                       # can still work (& is not DDPS)
            num_workers=4,  # number of distributed workers - this is really up to you
            use_gpu=True,    # whether each worker should use a gpu
            # NOTE: have to request more than 1 GPU explicitly in the slurm script
            resources_per_worker={"CPU": 1, "GPU": 0.5},   # should request 5 cpus, 2 gpus (+1 trainer)
            # with 1 worker, the above should run 4 concurrent trials on the same node
            # nodes are not easily available on Della - try to get everything in one node
            placement_strategy="PACK",   # PACK favors memory (locality) and SPREAD favors speed
            #trainer_resources={"CPU": 0},   # resources for the trainer (which i think is started on top of the workers, one for each node)
            # prob setting to 0 is the same if using 1 worker
        )
    
        run_config = RunConfig(           
        # checkpoint_config=CheckpointConfig(
        #     num_to_keep=3,   # TODO: what is this??
        #     checkpoint_score_attribute="loss",
        #     checkpoint_score_order="max",
        # ),
            storage_path=str(RUN_DIR),
            verbose=1,
            progress_reporter=CLIReporter(
                    metric_columns=["loss", "training_iteration"],
                    sort_by_metric=True,
                ),
        )
        ray_trainable = TorchTrainer(
                tune.with_parameters(train_parallel,run_tag=run_tag_in,path_dict=path_dict,config_dict=config_dict),  # BUG: check this passing
                scaling_config=scaling_config,
                run_config=run_config,
            )
    else:
        trainable = tune.with_parameters(train,run_tag=run_tag_in,path_dict=path_dict,config_dict=config_dict)
        # NOTE: since usually there is only 1 gpu available, and I've been requesting
        # this, it can only run 1 trial at a time. So the number of concurrent trials
        # is given by the number of GPUs (at least on Adroit/Della)
        # NOTE: you don't set workers if not trying to do DDPS - so num_workers (Ray actors) is 1
        ray_trainable = tune.with_resources(trainable, {'cpu': 1, 'gpu': 0.25})

    tuner_object = tune.Tuner(
        ray_trainable,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=1,    # TODO: should this be more? in case of grid search, just once, otherwise it repeats the same combination num_sample times
            search_alg=BasicVariantGenerator(),   # TODO: exploit others
            scheduler=FIFOScheduler(),    # TODO: exploit others
            trial_dirname_creator=trial_function,
        ),
        param_space={"train_loop_config": SEARCH_SPACE},   # TODO: is this right?
    )

    # start Ray Tune, can retrieve results with tuner_object.get_results()
    tuner_object.fit()

if __name__ == "__main__":
    main(
        run_tag_in=RUN_TAG,
        path_dict=path_dict,
        config_dict=config_dict,
    )
