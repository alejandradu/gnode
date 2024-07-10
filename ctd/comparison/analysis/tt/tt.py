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


# TODO: big description of this class and its methods


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
                #self.datamodule = torch.load(f, map_location=torch.device('cpu'))
                #self.datamodule = pickle.load(f)
                self.datamodule = torch.load(f, map_location=lambda storage, loc: storage.cpu())
                self.datamodule.prepare_data()
                self.datamodule.setup()
        else:
            with open(filepath + "datamodule_sim.pkl", "rb") as f:
                self.datamodule = pickle.load(f)
                self.datamodule.prepare_data()
                self.datamodule.setup()
        # self.env = self.datamodule.data_env.dataset_name
        # if the simulator exists
        if Path(filepath + "simulator.pkl").exists():
            with open(filepath + "simulator.pkl", "rb") as f:
                self.simulator = pickle.load(f)
        n_train = len(self.datamodule.train_ds)
        n_val = len(self.datamodule.valid_ds)
        n_test = len(self.datamodule.test_ds)
        self.n_trials = n_train + n_val + n_test
        # TODO: was the splitting really guaranteed to be in order?
        self.train_inds = range(0, int(0.8 * self.n_trials))
        self.valid_inds = range(int(0.8 * self.n_trials), self.n_trials)

    def get_inputs(self, phase="all"):
        train_ds = self.datamodule.train_ds
        valid_ds = self.datamodule.valid_ds
        tt_inputs = torch.cat([train_ds.tensors[1], valid_ds.tensors[1]], dim=0)
        if phase == "all":
            return tt_inputs
        elif phase == "train":
            return tt_inputs[self.train_inds]
        elif phase == "val":
            return tt_inputs[self.valid_inds]

    def get_true_inputs(self, phase="all"):
        train_ds = self.datamodule.train_ds
        valid_ds = self.datamodule.valid_ds
        tt_inputs = torch.cat([train_ds.tensors[7], valid_ds.tensors[7]], dim=0)
        if phase == "all":
            return tt_inputs
        elif phase == "train":
            return tt_inputs[self.train_inds]
        elif phase == "val":
            return tt_inputs[self.valid_inds]

    def get_inputs_to_env(self, phase="all"):
        if phase == "all":
            train_inputs_to_env = self.datamodule.train_ds.tensors[6]
            valid_inputs_to_env = self.datamodule.valid_ds.tensors[6]
            return torch.cat([train_inputs_to_env, valid_inputs_to_env], dim=0)
        elif phase == "train":
            return self.datamodule.train_ds.tensors[6]
        elif phase == "val":
            return self.datamodule.valid_ds.tensors[6]

    def get_model_inputs(self, phase="all"):

        if phase == "all":
            train_ics = self.datamodule.train_ds.tensors[0]
            train_inputs = self.datamodule.train_ds.tensors[1]
            train_targets = self.datamodule.train_ds.tensors[2]
            valid_ics = self.datamodule.valid_ds.tensors[0]
            valid_inputs = self.datamodule.valid_ds.tensors[1]
            valid_targets = self.datamodule.valid_ds.tensors[2]
            tt_ics = torch.cat([train_ics, valid_ics], dim=0)
            tt_inputs = torch.cat([train_inputs, valid_inputs], dim=0)
            tt_targets = torch.cat([train_targets, valid_targets], dim=0)
            return tt_ics, tt_inputs, tt_targets
        elif phase == "train":
            return (
                self.datamodule.train_ds.tensors[0],
                self.datamodule.train_ds.tensors[1],
                self.datamodule.train_ds.tensors[2],
            )
        elif phase == "val":
            return (
                self.datamodule.valid_ds.tensors[0],
                self.datamodule.valid_ds.tensors[1],
                self.datamodule.valid_ds.tensors[2],
            )

    def get_extra_inputs(self, phase="all"):
        if phase == "all":
            train_extra = self.datamodule.train_ds.tensors[5]
            valid_extra = self.datamodule.valid_ds.tensors[5]
            tt_extra = torch.cat([train_extra, valid_extra], dim=0)
            return tt_extra
        elif phase == "train":
            return self.datamodule.train_ds.tensors[5]
        elif phase == "val":
            return self.datamodule.valid_ds.tensors[5]

    def get_model_inputs_noiseless(self, phase="all"):
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs(phase=phase)

        train_noiseless_inputs = self.datamodule.train_ds.tensors[7]
        valid_noiseless_inputs = self.datamodule.valid_ds.tensors[7]
        tt_noiseless_inputs = torch.cat(
            [train_noiseless_inputs, valid_noiseless_inputs], dim=0
        )

        if phase == "all":
            return tt_ics, tt_noiseless_inputs, tt_targets
        elif phase == "train":
            return tt_ics, train_noiseless_inputs, tt_targets
        elif phase == "val":
            return tt_ics, valid_noiseless_inputs, tt_targets

    def get_model_outputs(self, phase="all"):
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs(phase=phase)
        out_dict = self.wrapper(tt_ics, tt_inputs, tt_targets)
        return out_dict

    def get_model_outputs_noiseless(self, phase="all"):
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs_noiseless(phase=phase)
        out_dict = self.wrapper(tt_ics, tt_inputs, tt_targets)
        return out_dict

    def get_latents(self, phase="all"):
        out_dict = self.get_model_outputs(phase=phase)
        return out_dict["latents"]

    def get_latents_noiseless(self, phase="all"):
        out_dict = self.get_model_outputs_noiseless(phase=phase)
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
        color_dict = {-1: 'red', 0: 'magenta', 1: 'green'}
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

        plt.subplots_adjust(hspace=1.5)  # adjust the space between subplots
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
        node=False,
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
        if hasattr(self.wrapper.model, "generator"):
            cell = self.wrapper.model.generator
        else:
            cell = self.wrapper.model.cell
        fps = find_fixed_points(
            model=cell,
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
        )
        return fps

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
        node=True,   # if True, use generator instance
    ):

        latents = self.get_latents(phase="val").detach().numpy()
        fps = self.compute_FPs(
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
            node=node,
        )
        xstar = fps.xstar
        q_vals = fps.qstar  
        is_stable = fps.is_stable
        figQs = plt.figure()
        axQs = figQs.add_subplot(111)
        q_flag_temp = q_vals < 1e-15
        q_vals[q_flag_temp] = 1e-15
        axQs.hist(np.log10(q_vals), bins=100)
        axQs.set_title("Q* Histogram")
        axQs.set_xlabel("log10(Q*)")

        colors = np.zeros((xstar.shape[0], 3))
        colors[is_stable, :] = np.array([0, 0, 1])
        colors[~is_stable, 0] = 0  # black

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
                            lats_pca[i, :, 2], linewidth=0.5,
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
                            latents[i, :, 2],linewidth=0.5,
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
        ax.plot([], [], "o", color="blue", label="Stable")
        ax.legend()
        ax.set_title("Fixed Points for Task-Trained")
        ax.set_xlabel("$m_1$")
        ax.set_ylabel("$m_2$")
        if xstar.shape[1] == 3:
            ax.set_zlabel("$m_3$")
        ax.set_facecolor('none')
        ax.grid(False)
        plt.show()
        
        if return_pca_model:
            return fps, pca
        
        if return_points:
            return fps, xstar, q_flag, colors
        
        return fps
    
    
    def plot_velocity_field_non_pca(analysis, input_field, latents_range, num_points, 
                                    xstar=None, q_flag=None, colors_fps=None, 
                                    num_traj=None, cmap=plt.cm.Reds, 
                                    input_trajectories=None, 
                                    scatter_trajectories=False, multiple_models=False):
        """
        Plot the flow field of the latent variables of a one or multiple
        low-dimensional RNN modelswithout PCA. Works for models with 
        latent size = 2, 3. Returns one plot overlaying the results, if
        multiple models are passed in the arguments.

        Args:
            analysis (list, Analysis_TT): list of Analysis_TT objects of one 
            or more models with the same latent dimension. Even if one model, 
            pass in as an array of one element.
            input_field (torch.tensor): input to calculate F at every point 
            in the grid (the field). 
            latents_range (list): list with each component the 
            coordinate limits for the grid defining the latent space to plot 
            the flow field.
            num_points (int): number of points along each dimension to get
            the grid.
            xstar (list): each element is an array specifying the fixed points 
            for the trajectories in each model. Objects returned by plot_fps()
            q_flag (list): each elements is an array with the same shape 
            as q_star, marking with 1's the valid fixed points. Object
            returned by plot_fps()
            colors_fps (list): each element is an array indicating the color
            code for the fixed points depending on their stability. 
            Object returned by plot_fps()
            num_traj (int): number of trajectories to plot
            over the flow field, obtained from the trials done during
            training (and with that input).
            cmap (plt.cm): colormap object to plot the flow field, depending
            on the magnitude of the velocity at every point.
            input_trajectories (list): each element is an array with 
            trajectories for a specific inputs. Especially useful when
            wanting to verify that the trajectories follow the flow
            field for a given input_field. Object returned by functions
            like get_latents()
            scatter_trajectories (bool, optional): if True, states visited
            by the trajectories are shown as points, apart from 
            connecting the trajectories with lines.
            multiple_models (bool, optional): if True, overlays then fields,
            and trajectories and fixed points (if given) with increasing
            red color intensity.
        """
        
        fig, ax = plt.subplots(figsize=(15, 15))
        
        if multiple_models:
            cmulti = plt.cm.Reds(np.linspace(0.35, 0.8, len(analysis))) 
        
        for index, a in enumerate(analysis):
            model = a.wrapper.model.generator
    
            # Calculate velocities over a grid using a double for loop implementation
            x = np.linspace(latents_range[0][0], latents_range[0][1], num_points)
            y = np.linspace(latents_range[1][0], latents_range[1][1], num_points)
            if len(latents_range) == 3:
                z = np.linspace(latents_range[2][0], latents_range[2][1], num_points)
                
            if len(latents_range) == 2:
                U = np.zeros([num_points, num_points])
                V = np.zeros([num_points, num_points])
            else:
                U = np.zeros([num_points, num_points, num_points])
                V = np.zeros([num_points, num_points, num_points])
                W = np.zeros([num_points, num_points, num_points])
                
            for i in range(num_points):
                for j in range(num_points):
                    state = torch.tensor([[x[i], y[j]]], dtype=torch.float)
                    if len(latents_range) == 2:
                        U[i, j], V[i, j] = (model(input_field, state) - state).detach().numpy().flatten()
                    else:
                        for k in range(num_points):
                            state = torch.tensor([[x[i], y[j], z[k]]], dtype=torch.float)
                            U[i, j, k], V[i, j, k], W[i, j, k] = 0.1*(model(input_field, state) - state).detach().numpy().flatten()
            
            # Create a colormap based on the normalized magnitude
            if len(latents_range) == 2:
                magnitude = np.sqrt(U**2 + V**2)
            else:
                magnitude = np.sqrt(U**2 + V**2 + W**2)
            normalized_magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
            colors_map = cmap(normalized_magnitude.flatten())
    
            # Plot the velocity field
            if len(latents_range) == 2:
                if multiple_models:
                    ax.quiver(*np.meshgrid(x, y, indexing='ij'), U, V, color=cmulti[index])
                else:
                    ax.quiver(*np.meshgrid(x, y, indexing='ij'), U, V, color=colors_map)
                #ax.quiver(*np.meshgrid(x, y, indexing='ij'), U, V)
            else:
                ax = fig.add_subplot(111, projection='3d')
                if multiple_models:  # scale up to magnify differences
                    ax.quiver(*np.meshgrid(x, y, z, indexing='ij'), U*100, V*100, W*100, color=cmulti[index])
                else:
                    ax.quiver(*np.meshgrid(x, y, z, indexing='ij'), U, V, W, color=colors_map)
            
            # plot trajectories
            if input_trajectories: 
                latents = a.get_custom_input_latents(input_trajectories)
            else:
                latents, colors_target, cmap_target = a.get_latents_noiseless_target_color()
                # latents = self.get_latents().detach().numpy()
            colors_time = plt.cm.viridis(np.linspace(0, 1, latents.shape[1]))
            for i in range(num_traj):
                if multiple_models:
                    ax.plot(*latents[i].T, linewidth=0.25, color=cmulti[index])
                else:
                    ax.plot(*latents[i].T, linewidth=0.25, color='black')
                ax.set_xlim(latents_range[0])
                ax.set_ylim(latents_range[1])
                if scatter_trajectories:
                    if input_trajectories:
                        if multiple_models:
                            ax.scatter(*latents[i].T, s=7, color=cmulti[index])
                        else:
                            ax.scatter(*latents[i].T, s=7)
                    else:
                        if multiple_models:
                            ax.scatter(*latents[i].T, s=7, color=cmulti[index])
                        else:
                            ax.scatter(*latents[i].T, s=7, c=colors_target[i], cmap=cmap_target)
            ax.set_xlim(latents_range[0])
            ax.set_ylim(latents_range[1])
                
            # plot fixed points
            if xstar[index] is not None and q_flag[index] is not None and colors_fps[index] is not None:
                xstar2 = xstar[index]
                colors2 = colors_fps[index]
                q_flag2 = q_flag[index]
                if multiple_models:   # still have to include difference in stability
                    ax.scatter(*xstar2[q_flag2].T, color=cmulti[index])
                else:
                    ax.scatter(*xstar2[q_flag2].T, c=colors2[q_flag2, :])
    
        ax.set_title("Latent Velocity Field")
        ax.set_ylabel("$m_2$", fontsize=25)
        ax.set_xlabel("$m_1$", fontsize=25)
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        if multiple_models:
            ax.set_zlabel("$m_3$", fontsize=25)
            ax.set_zlim(latents_range[2])
        plt.show()
    
    # TODO: maybe add a trajectory animation method
    
    # get trajectories where input is always value
    def get_custom_input_latents(self, value):
        ics, inputs, targets = self.get_model_inputs_noiseless()
        # create a new input tensor of the same shape wit value
        new_inputs = torch.ones(inputs.shape) * value
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
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        ax.plot(range(1, max_pcs + 1), pca.explained_variance_ratio_ * 100, marker="o")
        ax.set_xlabel("PC #")
        ax.set_title("Scree Plot")
        ax.set_ylabel("Explained Variance (%)")
        ax2 = fig.add_subplot(122)
        ax2.plot(range(1, max_pcs + 1), np.cumsum(pca.explained_variance_ratio_) * 100)
        ax2.set_xlabel("PC #")
        ax2.set_title("Cumulative Explained Variance")
        ax2.set_ylabel("Explained Variance (%)")
        # Add horiz lines at 50, 90, 95, 99%
        ax2.axhline(y=50, color="r", linestyle="--")
        ax2.axhline(y=90, color="r", linestyle="--")
        ax2.axhline(y=95, color="r", linestyle="--")
        ax2.axhline(y=99, color="r", linestyle="--")
        # Add y ticks
        ax2.set_yticks([50, 90, 95, 99])
        plt.savefig(f"{HOME_DIR}/scree_plot.png")
        return pca.explained_variance_ratio_
    
    def get_param(self, param, detach=True):
        """Get a specific parameter of the model after training"""
        if detach:
            return self.model.state_dict()[param].detach().numpy()
        else:
            return self.model.state_dict()[param]
        
    #def get_J_matrix(self):
        # wrt fitting neural data (and there is a firing rate lambda(t))