import os
import pickle
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import torch
from DSA.stats import dsa_bw_data_splits, dsa_to_id
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from ctd.comparison.analysis.analysis import Analysis
from ctd.comparison.fixedpoints import find_fixed_points

dotenv.load_dotenv(override=True)
HOME_DIR = os.getenv("HOME_DIR")


class Analysis_TT(Analysis):
    def __init__(self, run_name, filepath, use_train_dm=False):
        # initialize superclass
        super().__init__(run_name, filepath)
        self.tt_or_dt = "tt"
        self.load_wrapper(filepath, use_train_dm)
        self.run_hps = None

    def load_wrapper(self, filepath, use_train_dm=False):

        with open(filepath + "model.pkl", "rb") as f:
            # you pass in (initial conditions, inputs, targets) to wrapper
            self.wrapper = pickle.load(f)
        self.env = self.wrapper.task_env
        # you pass in (inputs, latent_state) to model.cell
        # so I am calculating the field for something with set inputs and targets
        self.model = self.wrapper.model
        if use_train_dm:
            with open(filepath + "datamodule_train.pkl", "rb") as f:
                self.datamodule = torch.load(f, map_location=torch.device('cpu'))
                #self.datamodule = pickle.load(f)
                self.datamodule.prepare_data()
                self.datamodule.setup()
        else:
            with open(filepath + "datamodule_sim.pkl", "rb") as f:
                self.datamodule = pickle.load(f)
                self.datamodule.prepare_data()
                self.datamodule.setup()
        self.task_name = self.datamodule.data_env.dataset_name
        # if the simulator exists
        if Path(filepath + "simulator.pkl").exists():
            with open(filepath + "simulator.pkl", "rb") as f:
                self.simulator = pickle.load(f)

    def get_inputs(self):
        all_data = self.datamodule.all_data
        tt_inputs = torch.Tensor(all_data["inputs"])
        return tt_inputs

    def get_inputs_to_env(self):
        return torch.Tensor(self.datamodule.all_data["inputs_to_env"])

    def get_model_inputs(self):
        all_data = self.datamodule.all_data
        tt_ics = torch.Tensor(all_data["ics"])
        tt_inputs = torch.Tensor(all_data["inputs"])
        tt_targets = torch.Tensor(all_data["targets"])
        return tt_ics, tt_inputs, tt_targets

    def get_extra_inputs(self):
        all_data = self.datamodule.all_data
        tt_extra = torch.Tensor(all_data["extra"])
        return tt_extra

    def get_model_inputs_noiseless(self):
        all_data = self.datamodule.all_data
        tt_ics = torch.Tensor(all_data["ics"])
        tt_inputs = torch.Tensor(all_data["true_inputs"])
        tt_targets = torch.Tensor(all_data["targets"])
        print()
        return tt_ics, tt_inputs, tt_targets

    def get_model_outputs(self):
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs()
        out_dict = self.wrapper(tt_ics, tt_inputs, tt_targets)
        return out_dict

    def get_model_outputs_noiseless(self):
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs_noiseless()
        out_dict = self.wrapper(tt_ics, tt_inputs, tt_targets)
        return out_dict
    
    def get_latents(self):
        out_dict = self.get_model_outputs()
        return out_dict["latents"]

    def get_latents_noiseless(self):
        out_dict = self.get_model_outputs_noiseless()
        return out_dict["latents"]
    
    def get_latents_noiseless_target_color(self):
        # return the latents and a color array depending on if the trajectory 
        # had a target of -1, 0 or 1
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs()
        out_dict = self.wrapper(tt_ics, tt_inputs, tt_targets)
        latents = out_dict["latents"].detach().numpy()
        tt_targets = tt_targets.detach().numpy()
        trials, steps, dimension = tt_targets.shape
        
        # Create a color array based on tt_targets
        color_dict = {-1: 'red', 0: 'green', 1: 'blue'}
        colors = np.array([[color_dict[tt_targets[t,s,0]] for s in range(steps)] for t in range(tt_targets.shape[0])])

        # Create a colormap using color_dict
        cmap = ListedColormap(['red', 'green', 'blue'])
             
        return latents, colors, cmap

    def get_latents_pca(self, num_PCs=3):
        latents = self.get_latents()
        B, T, N = latents.shape
        latents = latents.reshape(-1, N)
        pca = PCA(n_components=num_PCs)
        latents_pca = pca.fit_transform(latents)
        latents_pca = latents.reshape(B, T, num_PCs)
        return latents_pca, pca

    def plot_trial_latents(self, num_trials=10):
        out_dict = self.get_model_outputs()
        latents = out_dict["latents"].detach().numpy()
        pca = PCA(n_components=3)
        lats_pca = pca.fit_transform(latents.reshape(-1, latents.shape[-1]))
        lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 3)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for i in range(num_trials):
            ax.plot(
                lats_pca[i, :, 0],
                lats_pca[i, :, 1],
                lats_pca[i, :, 2],
            )
        ax.set_title("Task-trained Latent Activity")
        plt.show()

    def plot_trial_io(self, num_trials, n_pca_components=3):
        ics, inputs, targets = self.get_model_inputs()
        out_dict = self.get_model_outputs()
        latents = out_dict["latents"].detach().numpy()
        controlled = out_dict["controlled"].detach().numpy()
        pca = PCA(n_components=n_pca_components)
        lats_pca = pca.fit_transform(latents.reshape(-1, latents.shape[-1]))
        lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], n_pca_components)
        fig = plt.figure(figsize=(3 * num_trials, 6))

        for i in range(num_trials):
            ax1 = fig.add_subplot(4, num_trials, i + 1)
            for j in range(n_pca_components):
                ax1.plot(lats_pca[i, :, j])
            ax1.set_title(f"Trial {i}")

            ax2 = fig.add_subplot(4, num_trials, i + num_trials + 1)
            for j in range(controlled.shape[-1]):
                ax2.plot(controlled[i, :, j])

            ax3 = fig.add_subplot(4, num_trials, i + 2 * num_trials + 1)
            for j in range(targets.shape[-1]):
                ax3.plot(targets[i, :, j])

            ax4 = fig.add_subplot(4, num_trials, i + 3 * num_trials + 1)
            for j in range(inputs.shape[-1]):
                ax4.plot(inputs[i, :, j])

            if i == 0:
                ax1.set_ylabel("Latent Activity")
                ax2.set_ylabel("Controlled")
                ax3.set_ylabel("Targets")
                ax4.set_ylabel("Inputs")

            if i == 4:
                ax1.set_xlabel("Time")
                ax2.set_xlabel("Time")
                ax3.set_xlabel("Time")
                ax4.set_xlabel("Time")
            else:
                ax1.set_xlabel("")
                ax2.set_xlabel("")
                ax3.set_xlabel("")
                ax4.set_xlabel("")
                ax1.set_xticks([])
                ax2.set_xticks([])
                ax3.set_xticks([])
                ax4.set_xticks([])

        plt.suptitle("Task-trained Latent Activity")
        plt.show()
        
    def plot_trial_io_no_pca(self, num_trials, latent_size):
        ics, inputs, targets = self.get_model_inputs()
        out_dict = self.get_model_outputs()
        latents = out_dict["latents"].detach().numpy()
        controlled = out_dict["controlled"].detach().numpy()
        fig = plt.figure(figsize=(3 * num_trials, 6))

        for i in range(num_trials):
            ax1 = fig.add_subplot(4, num_trials, i + 1)
            for j in range(latent_size):
                ax1.plot(latents[i, :, j])
            ax1.set_title(f"Trial {i}")
            
            ax2 = fig.add_subplot(4, num_trials, i + num_trials + 1)
            for j in range(controlled.shape[-1]):
                ax2.plot(controlled[i, :, j])
                
            ax3 = fig.add_subplot(4, num_trials, i + 2 * num_trials + 1)
            for j in range(targets.shape[-1]):
                ax3.plot(targets[i, :, j])

            ax4 = fig.add_subplot(4, num_trials, i + 3 * num_trials + 1)
            for j in range(inputs.shape[-1]):
                ax4.plot(inputs[i, :, j])
                
            if i == 0:
                ax1.set_ylabel("Latent Activity")
                ax2.set_ylabel("Controlled")
                ax3.set_ylabel("Targets")
                ax4.set_ylabel("Inputs")

            if i == 4:
                ax1.set_xlabel("Time")
                ax2.set_xlabel("Time")
                ax3.set_xlabel("Time")
                ax4.set_xlabel("Time")
            else:
                ax1.set_xlabel("")
                ax2.set_xlabel("")
                ax3.set_xlabel("")
                ax4.set_xlabel("")
                ax1.set_xticks([])
                ax2.set_xticks([])
                ax3.set_xticks([])
                ax4.set_xticks([])

        plt.show()

    def compute_FPs(
        self,
        noiseless=True,
        inputs=None,
        n_inits=1024,
        noise_scale=0.0,
        learning_rate=1e-3,
        max_iters=10000,
        device="cpu",
        seed=0,
        compute_jacobians=True,
        report_progress = True,
        initial_states = None,
    ):
        # Compute latent activity from task trained model
        if inputs is None and noiseless:
            _, inputs, _ = self.get_model_inputs_noiseless()
            latents = self.get_latents_noiseless()
        elif inputs is None and not noiseless:
            _, inputs, _ = self.get_model_inputs()
            latents = self.get_latents()
        else:
            latents = self.get_latents()

        fps, velocity_vectors = find_fixed_points(
            model=self.wrapper.model.cell,
            state_trajs=latents,
            inputs=inputs,
            n_inits=n_inits,
            noise_scale=noise_scale,
            learning_rate=learning_rate,
            max_iters=max_iters,
            device=device,
            seed=seed,
            compute_jacobians=compute_jacobians,
            report_progress = report_progress,
            initial_states = initial_states,
        )
        return fps, velocity_vectors

    def plot_fps(
        self,
        inputs=None,
        num_traj=10,
        n_inits=1024,
        noise_scale=0.0,
        learning_rate=1e-3,
        max_iters=10000,
        device="cpu",
        seed=0,
        compute_jacobians=True,
        q_thresh=1e-5,
        n_pca_components=3,
        return_pca_model = False,
        do_pca=True,
        plot_only_points=False,
        report_progress = True,
        return_points = False,
        noiseless=True,
        initial_states = None,
    ):

        latents = self.get_latents().detach().numpy()
        fps, velocity_vectors = self.compute_FPs(
            noiseless=noiseless,
            inputs=inputs,
            n_inits=n_inits,
            noise_scale=noise_scale,
            learning_rate=learning_rate,
            max_iters=max_iters,
            device=device,
            seed=seed,
            compute_jacobians=compute_jacobians,
            report_progress = report_progress,
            initial_states = initial_states,
        )
        xstar = fps.xstar
        q_vals = fps.qstar
        is_stable = fps.is_stable
        figQs = plt.figure()
        axQs = figQs.add_subplot(111)
        axQs.hist(np.log10(q_vals), bins=100)
        axQs.set_title("Q* Histogram")
        axQs.set_xlabel("log10(Q*)")

        colors = np.zeros((xstar.shape[0], 3))
        colors[is_stable, :] = np.array([0, 0.3922, 0])  # darkgreen
        colors[~is_stable, 0] = 0

        q_flag = q_vals < q_thresh
        if do_pca:
            pca = PCA(n_components=n_pca_components)
            xstar_pca = pca.fit_transform(xstar)
            lats_flat = latents.reshape(-1, latents.shape[-1])
            lats_pca = pca.transform(lats_flat)

            if n_pca_components == 3:
                lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 3)
                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                   xstar_pca[q_flag, 0],
                   xstar_pca[q_flag, 1],
                   xstar_pca[q_flag, 2],
                   c=colors[q_flag, :]
                )
                if not plot_only_points:
                    for i in range(num_traj):
                        ax.plot(
                            lats_pca[i, :, 0],
                            lats_pca[i, :, 1],
                            lats_pca[i, :, 2],
                        )
            elif n_pca_components == 2:
                lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 2)
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.scatter(
                   xstar_pca[q_flag, 0],
                   xstar_pca[q_flag, 1],
                   c=colors[q_flag, :]
                )
                if not plot_only_points:
                    for i in range(num_traj):
                        ax.plot(
                            lats_pca[i, :, 0],
                            lats_pca[i, :, 1],
                        )
                    
        else:
            if xstar.shape[1] == 3:
                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                   xstar[q_flag, 0],
                   xstar[q_flag, 1],
                   xstar[q_flag, 2],
                   c=colors[q_flag, :]
                )
                if not plot_only_points:
                    for i in range(num_traj):
                        ax.plot(
                            latents[i, :, 0],
                            latents[i, :, 1],
                            latents[i, :, 2],
                        )
            elif xstar.shape[1] == 2:
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.scatter(
                   xstar[q_flag, 0],
                   xstar[q_flag, 1],
                   c=colors[q_flag, :]
                )
                if not plot_only_points:
                    for i in range(num_traj):
                        ax.plot(
                            latents[i, :, 0],
                            latents[i, :, 1],
                        )
        
        # Add legend for stability
        ax.plot([], [], "o", color="black", label="Unstable")
        ax.plot([], [], "o", color="darkgreen", label="Stable")
        ax.legend()
        ax.set_title("Fixed Points for Task-Trained")
        plt.show()
        
        if return_pca_model:
            return fps, pca
        
        if return_points:
            return velocity_vectors, xstar, q_flag, colors
        
        return fps
        
    
    def plot_velocity_field_non_pca(self, inputs, latents_range, num_points, 
                                    xstar=None, q_flag=None, colors=None, 
                                    num_traj=None, cmap=plt.cm.Reds, 
                                    return_animation=False, validate_input_val=None):
        model = self.wrapper.model.cell

        # Prepare the state space based on the dimensions in latents_range
        grid = [np.linspace(range_[0], range_[1], num_points) for range_ in latents_range]
        mesh = np.meshgrid(*grid, indexing='ij')
        states = np.stack([ax.ravel() for ax in mesh], axis=-1)

        # Convert numpy array to PyTorch tensor
        states_tensor = torch.from_numpy(states).float()

        # Calculate F for all states
        F = model(inputs, states_tensor)  # Assuming model takes only states_tensor as input

        # Compute velocity vectors
        delta = (F - states_tensor).detach().numpy()
        if len(latents_range) == 2:
            U, V = delta.T
        else:
            U, V, W = delta.T

        # Create a colormap based on the normalized magnitude
        magnitude = np.sqrt(np.sum(delta**2, axis=-1))
        normalized_magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
        colors_map = cmap(normalized_magnitude.flatten())

        # Plot the velocity field
        fig, ax = plt.subplots(figsize=(10, 10))
        if len(latents_range) == 2:
            ax.quiver(*np.meshgrid(*grid, indexing='ij'), U.reshape(num_points, num_points), V.reshape(num_points, num_points), color=colors_map)
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(*np.meshgrid(*grid, indexing='ij'), W.reshape(num_points, num_points, num_points), V.reshape(num_points, num_points, num_points), U.reshape(num_points, num_points, num_points), color=colors_map)
            
        # plot trajectories
        if validate_input_val is not None: 
            latents = self.get_empty_input_latents(validate_input_val)
        else:
            latents, colors_target, cmap_target = self.get_latents_noiseless_target_color()
            # latents = self.get_latents().detach().numpy()
        colors_time = plt.cm.viridis(np.linspace(0, 1, latents.shape[1]))
        for i in range(num_traj):
            ax.plot(*latents[i].T, linewidth=0.1, color='black')
            ax.set_xlim(latents_range[0])
            ax.set_ylim(latents_range[1])
            ax.scatter(*latents[i].T, c=colors_target[i], cmap=cmap_target)
        ax.set_xlim(latents_range[0])
        ax.set_ylim(latents_range[1])
        # ax.legend(handles=handles)
        
            
        # plot fixed points
        if xstar is not None and q_flag is not None and colors is not None:
            ax.scatter(*xstar[q_flag].T, c=colors[q_flag, :])

        ax.set_title("Latent Velocity Field")
        plt.show()
        
        if return_animation:
            return fig, ax
        
    
    def plot_velocity_field_pca(self, inputs, latents_range, num_points, 
                                    xstar=None, q_flag=None, colors=None, 
                                    num_traj=None, cmap=plt.cm.Reds, 
                                    return_animation=False, validate_input_val=None):
        pass
        
        

    # TODO: probably refactor to get the values of the velocity field
    # and plots separately
    def animate_trajectory(self, i, latents_range, num_points, inputs, filename='trajectory_animation.mp4'):
        # Get the i'th latent trajectory
        latents = self.get_latents().detach().numpy()
        trajectory = latents[i]

        # Create a figure and axes
        fig, ax = self.plot_velocity_field_non_pca(inputs, latents_range, num_points)

        # Create a scatter plot for the trajectory point
        S = ax.scatter(*trajectory[0], color='red')

        # Animation update function
        def update(num):
            # Update the trajectory point
            S.set_offsets(trajectory[num])

            # Return the updated artists
            return S,

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=200, blit=True)

        # Save the animation as a video file
        ani.save(filename, writer='ffmpeg')

        plt.close(fig)
    
    # get trajectories where input is always value
    def get_empty_input_latents(self, value):
        ics, inputs, targets = self.get_model_inputs_noiseless()
        # create a new input tensor of the same shape wit value
        new_inputs = torch.zeros(inputs.shape) * value
        # get the outputs
        out_dict = self.wrapper(ics, new_inputs, targets)
        return out_dict["latents"].detach().numpy()
        

    def simulate_neural_data(self, subfolder, dataset_path):
        self.simulator.simulate_neural_data(
            self.wrapper,
            self.datamodule,
            self.run_name,
            subfolder,
            dataset_path,
            seed=0,
        )

    def find_DSA_hps(
        self,
        rank_sweep=[10, 20],
        delay_sweep=[1, 5],
    ):
        id_comp = np.zeros((len(rank_sweep), len(delay_sweep)))
        splits_comp = np.zeros((len(rank_sweep), len(delay_sweep)))
        latents = self.get_latents().detach().numpy()
        latents = latents.reshape(-1, latents.shape[-1])
        for i, rank in enumerate(rank_sweep):
            for j, delay in enumerate(delay_sweep):
                print(f"Rank: {rank}, Delay: {delay}")
                id_comp[i, j] = dsa_to_id(
                    data=latents,
                    rank=rank,
                    n_delays=delay,
                    delay_interval=1,
                )
                splits_comp[i, j] = dsa_bw_data_splits(
                    data=latents,
                    rank=rank,
                    n_delays=delay,
                    delay_interval=1,
                )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(id_comp)
        ax.set_title("ID")
        ax.set_xticks(np.arange(len(delay_sweep)))
        ax.set_yticks(np.arange(len(rank_sweep)))
        ax.set_xticklabels(delay_sweep)
        ax.set_yticklabels(rank_sweep)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        plt.savefig(f"{HOME_DIR}/id_comp.png")
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.imshow(splits_comp)
        ax2.set_title("Splits")
        ax2.set_xticks(np.arange(len(delay_sweep)))
        ax2.set_yticks(np.arange(len(rank_sweep)))
        ax2.set_xticklabels(delay_sweep)
        ax2.set_yticklabels(rank_sweep)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.savefig(f"{HOME_DIR}/splits_comp.png")
        return id_comp, splits_comp

    def save_latents(self, filepath):
        latents = self.get_latents().detach().numpy()
        with open(filepath, "wb") as f:
            pickle.dump(latents, f)

    def plot_scree(self, max_pcs=10):
        latents = self.get_latents().detach().numpy()
        latents = latents.reshape(-1, latents.shape[-1])
        pca = PCA(n_components=max_pcs)
        pca.fit(latents)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(1, max_pcs + 1), pca.explained_variance_ratio_ * 100, marker="o")
        ax.set_xlabel("PC #")
        ax.set_title("Scree Plot")
        ax.set_ylabel("Explained Variance (%)")
        plt.savefig(f"{HOME_DIR}/scree_plot.png")
        return pca.explained_variance_ratio_

    def explore_params(self):
        _, inputs, _ = self.get_model_inputs_noiseless()
        print("shape of the inputs to model F: ", inputs.shape)
        latents = self.get_latents_noiseless()
        print("shape of the latents: ", latents.shape)
    