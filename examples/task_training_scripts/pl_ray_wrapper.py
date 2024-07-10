# This script integrates the pl lightning train module with the ray 
# tune hyperparameter tuning module 

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
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from utils import make_data_tag, trial_function

def send_task_train(model: str,
                    task: str,
                    run_desc: str, 
                    search_space,   # TODO: type hint
                    num_hyperparam_samples=1,
                    max_epochs=500,
                    tune_search_alg=BasicVariantGenerator(),
                    tune_scheduler=FIFOScheduler(),
                    local_mode=False,
                    overwrite=True,          
                    wandb_logging=False,
                    checkpoint_config: ray.train.CheckpointConfig = None,
                    resources_per_trial={"CPU": 1, "GPU": 0.5},
                    DDP=False,
                    DDP_num_workers=2,
                    DDP_use_gpu=True,
                    DDP_resources_per_worker={"CPU": 1, "GPU": 0.5},
                    DDP_placement_strategy="PACK",
                    DDP_assign_trainer_CPU=False,
                    ):
    """
    Send a task training job on a cluster.
    ctrl+c stops the training

    Args:
        model (str): The model to train. Choose from configs.
        task (str): The task to train on. Choose from configs.
        run_desc (str): Name tag of the run to save data in scratch.
        search_space (Dict[str, Dict[str, Any]]): The search space for 
        hyperparameter tuning using tune functions
        num_hyperparam_samples (int, optional): The number of hyperparameter
        combinations to sample from the tune search space. 1
        if just using grid search. Defaults to 1.
        max_epochs (int, optional): The maximum number of epochs. 
        Defaults to 500.
        tune_search_alg (BasicVariantGenerator, optional): Algorithm
        to sweep the hyperparameter space. Defaults to BasicVariantGenerator().
        tune_scheduler (FIFOScheduler, optional): The scheduler for 
        trials with different hyperparams. Defaults to FIFOScheduler().
        local_mode (bool, optional): Whether to run locally for debugging. 
        Defaults to False.
        overwrite (bool, optional): Whether to overwrite existing runs. 
        Defaults to True.
        wandb_logging (bool, optional): Whether to log to WandB. 
        Defaults to False.
        checkpoint_config (ray.train.CheckpointConfig, optional): 
        The checkpoint configuration. Defaults to None.
        resources_per_trial (Dict[str, float], optional): The resources
        per trial (not running DDP, only 1 implicit Ray worker). 
        Defaults to {"CPU": 1, "GPU": 0.5}.
        DDP (bool, optional): Whether to use Distributed Data Parallel 
        (DDP) for training. Defaults to False. NOTE that DDP needs a
        specific slurm script to assign the nodes.
        DDP_num_workers (int, optional): The number of ray workers for DDP.
        Different from num_workers that load the data in DataLoader.
        Defaults to 2.
        DDP_use_gpu (bool, optional): Whether to use GPU for DDP.
        Defaults to True.
        DDP_resources_per_worker (Dict[str, float], optional): The resources
        per worker for DDP. Defaults to {"CPU": 1, "GPU": 0.5}.
        DDP_placement_strategy (str, optional): The placement strategy 
        for DDP across nodes.Defaults to "PACK".
        DDP_assign_trainer_CPU (bool, optional): Whether to assign a CPU
        to the trainer. Defaults to False.
        
    """

    # Add custom resolver to create the data_tag so it can be used for run dir
    OmegaConf.register_new_resolver("make_data_tag", make_data_tag)
    log = logging.getLogger(__name__)

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

    date_str = datetime.now().strftime("%Y%m%d")
    run_tag = f"{date_str}_{run_desc}"
    run_dir = HOME_DIR / "content" / "runs" / "task-trained" / run_tag

    # -----------------Default Parameter Sets -----------------------------------
    config_dict = dict(
        task_wrapper=Path(f"configs/task_wrapper/{task}.yaml"),
        env_task=Path(f"configs/env_task/{task}.yaml"),
        env_sim=Path(f"configs/env_sim/{task}.yaml"),
        datamodule_task=Path(f"configs/datamodule_train/datamodule_{task}.yaml"),
        datamodule_sim=Path(f"configs/datamodule_sim/datamodule_{task}.yaml"),
        model=Path(f"configs/model/{model}.yaml"),
        simulator=Path(f"configs/simulator/default_{task}.yaml"),
        callbacks=Path(f"configs/callbacks/default_{task}.yaml"),
        loggers=Path("configs/logger/default.yaml"),
        trainer=Path("configs/trainer/default.yaml"),
    )

    if not wandb_logging:
        config_dict["loggers"] = Path("configs/logger/default_no_wandb.yaml")
        config_dict["callbacks"] = Path("configs/callbacks/default_no_wandb.yaml")

    if local_mode:
        ray.init(local_mode=True)
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
        
    run_dir.mkdir(parents=True)
    shutil.copyfile(__file__, run_dir / Path(__file__).name)
    
    # setup the run configuration
    run_config = RunConfig(
        checkpoint_config=checkpoint_config,
        #stop={"training_iteration": max_epochs},
        storage_path=str(run_dir),
        verbose=1,
        progress_reporter=CLIReporter(
            metric_columns=["loss", "training_iteration"],
            sort_by_metric=True,
        ),
    )

    if DDP:
        
        from ctd.task_modeling.task_train_prep_parallel import train_parallel
        
        scaling_config = ScalingConfig(    
            num_workers=DDP_num_workers, 
            use_gpu=DDP_use_gpu,  
            resources_per_worker=DDP_resources_per_worker,  
            placement_strategy=DDP_placement_strategy,   
            trainer_resources={"CPU": 1} if DDP_assign_trainer_CPU else {"CPU": 0},
        )
        
        ray_trainable = TorchTrainer(
                tune.with_parameters(train_parallel,run_tag=run_tag,path_dict=path_dict,config_dict=config_dict),  
                scaling_config=scaling_config,
                run_config=run_config,
        )
        
    else:
        
        from ctd.task_modeling.task_train_prep import train

        # trainable = tune.with_parameters(train,run_tag=run_tag,path_dict=path_dict,config_dict=config_dict)
        # ray_trainable = tune.with_resources(trainable, resources_per_trial)
        
        trainable = tune.with_parameters(train,run_tag=run_tag,path_dict=path_dict,config_dict=config_dict)
        ray_trainable = TorchTrainer(
                tune.with_resources(trainable, resources_per_trial),  
                run_config=run_config,
        )
        
    # gather configurations to train with custom hyperparam tune algorithms
    tuner_object = tune.Tuner(
        ray_trainable,
        param_space={"train_loop_config": search_space},  
        #run_config=run_config,  NOTE: it is already in TorchTrainer
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=num_hyperparam_samples,  
            search_alg=tune_search_alg,   
            scheduler=tune_scheduler,    
            trial_dirname_creator=trial_function,
        ),
    )

    # start Ray Tune, can retrieve results with tuner_object.get_results()
    tuner_object.fit()
