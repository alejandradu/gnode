from data_modeling.lfads_torch.datamodules import BasicDataModule
class TaskTrainWrapper():
    def __init__(self, task_env, model, data_simulator, trainer):
        self.model = model
        self.trainer = trainer
        self.datamodule = None
        self.seed = 0
    
    def create_datamodule(self):
        self.datamodule = BasicDataModule(
            n_samples = self.task_env.n_samples,
            n_timesteps = self.task_env.n_timesteps,
            )
        
    def tune_model(self):
        if self.datamodule is None:
            self.create_datamodule()
        self.trainer.tune(model = self.model, datamodule = self.datamodule)

    def train_model(self):
        if self.datamodule is None:
            self.create_datamodule()
        self.trainer.fit(model = self.model, datamodule = self.datamodule)

    def simulate_neural_spiking(self):
        # Pass the latent activity to the data simulator to generate spiking data
        self.data_simulator.simulate_neural_data(
            task_trained_model = self.model, 
            datamodule = self.datamodule,
            seed = self.seed
            )
    def train_and_simulate(self):
        self.train_model()
        self.simulate_neural_spiking()