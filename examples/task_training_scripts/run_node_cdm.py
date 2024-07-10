import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import dotenv
import ray
from omegaconf import OmegaConf
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator

from ctd.task_modeling.task_train_prep import train
from utils import make_data_tag, trial_function

# Add custom resolver to create the data_tag so it can be used for run dir
OmegaConf.register_new_resolver("make_data_tag", make_data_tag)
log = logging.getLogger(__name__)
dotenv.load_dotenv(override=True)

# ---------------Options---------------
LOCAL_MODE = False  # Set to True to run locally (for debugging)
OVERWRITE = True  # Set to True to overwrite existing run
WANDB_LOGGING = False  # Set to True to log to WandB (need an account)

MAX_EPOCHS = 300
MODEL = "NODE"
TASK = "MultiTask"
task_detail = "CDM_1"   
NUM_HYPERPARAM_SAMPLES = 1
RESOURCES_PER_TRIAL = {"CPU": 1, "GPU": 0.25}   # around 0.25 gpu per node_cdm trial
# TUNE_SEARCH_ALG=
# TUNE_SCHEDULER=

RUN_DESC = f"{MODEL}_{TASK}_{task_detail}_{MAX_EPOCHS}epoch"  # For WandB and run dir

# ----------------- Parameter Selection -----------------------------------

SEARCH_SPACE = dict(
    model = dict(
        latent_size = tune.grid_search([2,3,5,10]),
        # gating_linear = tune.grid_search([True]), 
        # gating_n_layers = tune.grid_search([1,10])    ,  
        # euler_step_size = tune.grid_search([0.1, 0.01]),
    ),
    # train loop configs
    task_wrapper=dict(
        weight_decay=1e-9,
        learning_rate=1e-2,
    ),
    # pl.Trainer configs
    trainer=dict(
        max_epochs=MAX_EPOCHS,  
        log_every_n_steps=25,            # steps != epochs
    ),
    # configs for dataset (timeseries trials) creation
    params=dict(
        seed=0,
        batch_size=32,   
        n_samples=800,                   # number of trials to simulate
        # num_workers=tune.choice([4]),  # for the DataLoader. != Ray workers. Usually set = CPUs
    ),
    # task parameters
    env_task=dict(
        task_list=["ContextIntMod1"],    # if using choice remember to use double brackets [[...]]
    ),
    # simulation task params
    env_sim=dict(
        task_list=["ContextIntMod1"],
    ),
)

# -------------------------- SLURM configs --------------------------
# If not running DDP:

# ntasks-per-node = total number of hyperparam combinations = trials 
# cpus-per-task = CPU count in resources per trial 
# gres:gpu = GPU count in resources per trial * trials

# If running DDP:

# TBD

# ------------------Data Management Variables --------------------------------

HOME_DIR = Path(os.environ.get("HOME_DIR"))
print(f"Saving files to {HOME_DIR}")

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
    tune.run(
        tune.with_parameters(
            train,
            run_tag=run_tag_in,
            path_dict=path_dict,
            config_dict=config_dict,
        ),
        metric="loss",
        mode="min",
        config=SEARCH_SPACE,
        resources_per_trial=RESOURCES_PER_TRIAL,
        num_samples=NUM_HYPERPARAM_SAMPLES,
        storage_path=str(RUN_DIR),
        search_alg=BasicVariantGenerator(),
        scheduler=FIFOScheduler(),
        verbose=1,
        progress_reporter=CLIReporter(
            metric_columns=["loss", "training_iteration"],
            sort_by_metric=True,
        ),
        trial_dirname_creator=trial_function,
    )


if __name__ == "__main__":
    main(
        run_tag_in=RUN_TAG,
        path_dict=path_dict,
        config_dict=config_dict,
    )