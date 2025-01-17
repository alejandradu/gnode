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

RUN_DESC = "LintRNN_CDM"  # For WandB and run dir
TASK = "MultiTask"  # Task to train on (see configs/task_env for options)
MODEL = "DriscollRNN"  # Model to train (see configs/model for options)

# -----------------Parameter Selection ---------------------------------
SEARCH_SPACE = dict(
    model = dict(
        latent_size = tune.grid_search([512]),
        noise_level = tune.grid_search([0]),
        #l2_wt = tune.grid_search([1e-6, 1e-8]),
        # rank = tune.grid_search([2]),  only fits if rank != latent_size
        # noise level=0.05, gamma=1 (dt = tau)
    ),
    task_wrapper=dict(
        # Task Wrapper Parameters - high learning rate, low weight decay
        weight_decay=tune.grid_search([1e-8]), # does not make much difference
        learning_rate=tune.grid_search([1e-2]),
    ),
    trainer=dict(
        # Trainer Parameters 
        max_epochs=tune.choice([10]),
        log_every_n_steps=tune.choice([1]),   # rather logs every epoch
    ),
    # Data Parameters 
    params=dict(
        seed=tune.grid_search([0]),
        batch_size=tune.choice([24]),
        num_workers=tune.choice([1]),
        n_samples=tune.choice([1000]),  # number of trials to simulate
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
        mode="min", 
        config=SEARCH_SPACE,
        resources_per_trial=dict(cpu=1, gpu=1), 
        num_samples=1,                      # HERE: number of times to sample from hyperparam space
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
