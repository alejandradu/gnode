from ray import tune
from examples.task_training_scripts.pl_ray_wrapper import send_task_train


# This script launches a task training job 
# There will be more/less fields depending on the model and task

MAX_EPOCHS = 3
MODEL = "NODE"
TASK = "MultiTask"
task_detail = "CDM_mini"   
NUM_HYPERPARAM_SAMPLES = 1
RESOURCES_PER_TRIAL = {"CPU": 1, "GPU": 0.5}
# tune_search_alg=
# tune_scheduler=
# local_mode=False,
# overwrite=True,
# wandb_logging=False,
# DDP=False,
# DDP_num_workers=2,
# DDP_use_gpu=True,
# DDP_resources_per_worker={"CPU": 1, "GPU": 0.5},
# DDP_placement_strategy="PACK",
# DDP_assign_trainer_CPU=False,

SEARCH_SPACE = dict(
    model = dict(
        latent_size = tune.grid_search([2,3]),
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

# ---------------------------- send job -----------------------------

send_task_train(
    model=MODEL,
    task=TASK,
    run_desc=f"{MODEL}_{TASK}_{task_detail}_{MAX_EPOCHS}epoch",
    search_space=SEARCH_SPACE,
    num_hyperparam_samples=NUM_HYPERPARAM_SAMPLES,
    max_epochs=MAX_EPOCHS,
    resources_per_trial=RESOURCES_PER_TRIAL,
)
