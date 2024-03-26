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

from ctd.task_modeling.task_train_prep import train
from utils import make_data_tag, trial_function

# Add custom resolver to create the data_tag so it can be used for run dir
OmegaConf.register_new_resolver("make_data_tag", make_data_tag)
log = logging.getLogger(__name__)

# ---------------Options---------------
LOCAL_MODE = False  # Set to True to run locally (for debugging)
OVERWRITE = True  # Set to True to overwrite existing run
WANDB_LOGGING = False  # Set to True to log to WandB (need an account)

RUN_DESC = "NODE_N3BFF_trial"  # For WandB and run dir
TASK = "NBFF"  # N=3, Task to train on (see configs/task_env for options)
MODEL = "NODE"  # Model to train (see configs/model for options)

# -----------------Parameter Selection -----------------------------------
SEARCH_SPACE = dict(
    # model = dict(
    #     latent_size = tune.grid_search([4]),
    # ),
    task_wrapper=dict(
        # Task Wrapper Parameters -----------------------------------
        # the 2 below are really the only ones that should be tuned
        weight_decay=tune.grid_search([1e-8]),
        learning_rate=tune.grid_search([1e-3]),
    ),
    trainer=dict(
        # Trainer Parameters -----------------------------------
        max_epochs=tune.choice([10]),
    ),
    # Data Parameters -----------------------------------
    params=dict(
        seed=tune.grid_search([0]),
        batch_size=tune.choice([256]),
        num_workers=tune.choice([4]),
        n_samples=tune.choice([10]),
        # data_env (Env): The environment to simulate
        #     n_samples (int): The number of samples (trials) to simulate
        #     seed (int): The random seed to use
        #     batch_size (int): The batch size
        #     num_workers (int): The number of workers to use for data loading
    ),
)

# add batch size, learning rate, etc


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
        tune.with_parameters(train,run_tag=run_tag_in,path_dict=path_dict,config_dict=config_dict),
        metric="loss",
        mode="min",  # minimize the hyperparameter
        config=SEARCH_SPACE,
        resources_per_trial=dict(cpu=2, gpu=1),  # try doubling the cpu, check memory utilization
                                                   # nodes: number of servers 
                                                   # check number of trials
                                                   # or maybe increase the number of trials 
        num_samples=1,  # number of times to sample from hyperparam space
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