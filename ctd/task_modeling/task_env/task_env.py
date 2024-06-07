# Class to generate training data for task-trained RNN that does 3 bit memory task
from abc import ABC, abstractmethod

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from motornet.environment import Environment
from numpy import ndarray
from torch._tensor import Tensor
import textwrap

from ctd.task_modeling.task_env.loss_func import NBFFLoss

class DecoupledEnvironment(gym.Env, ABC):
    """
    Abstract class representing a decoupled environment.
    This class is abstract and cannot be instantiated.

    """

    # All decoupled environments should have
    # a number of timesteps and a noise parameter
    # TODO: the rest of the methods actually require more: 
    # action_space, observation_space, context_inputs, state_label, input_labels, output_labels, coupled_env, loss_func

    @abstractmethod
    def __init__(self, n_timesteps: int, noise: float):
        super().__init__()
        self.dataset_name = "DecoupledEnvironment"
        self.n_timesteps = n_timesteps
        self.noise = noise

    # All decoupled environments should have
    # functions to reset, step, and generate trials
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def generate_dataset(self, n_samples):
        """Must return a dictionary with the following keys:
        #----------Mandatory keys----------
        ics: initial conditions
        inputs: inputs to the environment
        targets: targets of the environment
        conds: conditions information (if applicable)
        extra: extra information
        #----------Optional keys----------
        true_inputs: true inputs to the environment (if applicable)
        true_targets: true targets of the environment (if applicable)
        phase_dict: phase information (if applicable)
        """

        pass


class NBitFlipFlop(DecoupledEnvironment):
    """
    An environment for an N-bit flip flop.
    This is a simple toy environment where the goal is to flip the required bit.
    """

    def __init__(
        self,
        n_timesteps: int,
        noise: float,
        n=1,
        switch_prob=0.01,
        transition_blind=4,
        dynamic_noise=0,
    ):
        super().__init__(n_timesteps=n_timesteps, noise=noise)
        self.dataset_name = f"{n}BFF"
        self.action_space = spaces.Box(low=-0.5, high=1.5, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(n,), dtype=np.float32
        )
        self.context_inputs = spaces.Box(
            low=-1.5, high=1.5, shape=(0,), dtype=np.float32
        )
        self.n = n
        self.state = np.zeros(n)
        self.input_labels = [f"Input {i}" for i in range(n)]
        self.output_labels = [f"Output {i}" for i in range(n)]
        self.noise = noise
        self.dynamic_noise = dynamic_noise
        self.coupled_env = False
        self.switch_prob = switch_prob
        self.transition_blind = transition_blind
        self.loss_func = NBFFLoss(transition_blind=transition_blind)

    def step(self, action):
        # Generates state update given an input to the flip-flop
        for i in range(self.n):
            if action[i] == 1:
                self.state[i] = 1
            elif action[i] == -1:
                self.state[i] = 0

    def generate_trial(self):
        # Make one trial of flip-flop
        self.reset()

        # Generate the times when the bit should flip
        inputRand = np.random.random(size=(self.n_timesteps, self.n))
        inputs = np.zeros((self.n_timesteps, self.n))
        inputs[
            inputRand > (1 - self.switch_prob)
        ] = 1  # 2% chance of flipping up or down
        inputs[inputRand < (self.switch_prob)] = -1

        # Set the first 3 inputs to 0 to make sure no inputs come in immediately
        inputs[0:3, :] = 0

        # Generate the desired outputs given the inputs
        outputs = np.zeros((self.n_timesteps, self.n))
        for i in range(self.n_timesteps):
            self.step(inputs[i, :])
            outputs[i, :] = self.state

        # Add noise to the inputs for the trial
        true_inputs = inputs
        inputs = inputs + np.random.normal(loc=0.0, scale=self.noise, size=inputs.shape)
        return inputs, outputs, true_inputs

    def reset(self):
        self.state = np.zeros(self.n)
        return self.state

    def generate_dataset(self, n_samples):
        # Generates a dataset for the NBFF task
        n_timesteps = self.n_timesteps
        ics_ds = np.zeros(shape=(n_samples, self.n))
        outputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.n))
        inputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.n))
        true_inputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.n))
        for i in range(n_samples):
            inputs, outputs, true_inputs = self.generate_trial()
            outputs_ds[i, :, :] = outputs
            inputs_ds[i, :, :] = inputs
            true_inputs_ds[i, :, :] = true_inputs

        dataset_dict = {
            "ics": ics_ds,
            "inputs": inputs_ds,
            "inputs_to_env": np.zeros(shape=(n_samples, n_timesteps, 0)),
            "targets": outputs_ds,
            "true_inputs": true_inputs_ds,
            "conds": np.zeros(shape=(n_samples, 1)),
            # No extra info for this task, so just fill with zeros
            "extra": np.zeros(shape=(n_samples, 1)),
        }
        extra_dict = {}
        return dataset_dict, extra_dict

    def render(self):
        inputs, states, _ = self.generate_trial()
        fig1, axes = plt.subplots(nrows=self.n + 1, ncols=1, sharex=True)
        colors = plt.cm.rainbow(np.linspace(0, 1, self.n))
        for i in range(self.n):
            axes[i].plot(states[:, i], color=colors[i])
            axes[i].set_ylabel(f"State {i}")
            axes[i].set_ylim(-0.2, 1.2)
        ax2 = axes[-1]
        for i in range(self.n):
            ax2.plot(inputs[:, i], color=colors[i])
        ax2.set_ylim(-1.2, 1.2)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Inputs")
        plt.figure(dpi=500)
        #plt.tight_layout(pad=20.0)  # increase padding to 2.0
        plt.show()
        #fig1.savefig("nbitflipflop.pdf")

    def render_3d(self, n_trials=10):
        if self.n > 2:
            fig = plt.figure(figsize=(5 * n_trials, 5))
            # Make colormap for the timestep in a trial
            for i in range(n_trials):

                ax = fig.add_subplot(1, n_trials, i + 1, projection="3d")
                inputs, states, _ = self.generate_trial()
                ax.plot(states[:, 0], states[:, 1], states[:, 2])
                ax.set_xlabel("Bit 1")
                ax.set_ylabel("Bit 2")
                ax.set_zlabel("Bit 3")
                ax.set_title(f"Trial {i+1}")
            plt.tight_layout()
            plt.show()


class OneBitSum(DecoupledEnvironment):
    """
    An environment for a 2-bit memory task.
    The output bit is the sign of the sum of all previous pulses.
    
    noise: float, the standard deviation of the noise to add to the inputs
    switch_prob: float, the probability of a bit flipping (if not Poisson, like NBFF)
    transition_blind: int, the number of timesteps to ignore after a transition
    limits: list, the limits of the sum (default is [-1, 1])
    n: int, the number of pulses to send in a trial (average number of poisson)
    poisson: bool, whether to use a poisson process for the pulses (NBFF bit flipping otherwise)
    
    """
    
    def __init__(self, n_timesteps: int, noise: float, switch_prob=0.05, transition_blind=1, n=1, limits = [-1, 1], poisson=True):
        
        super().__init__(n_timesteps=n_timesteps, noise=noise)
        self.dataset_name = "OneBitSum"
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.limits = limits
        # this has always been in the range -1, 1
        if limits is not None:
            self.observation_space = spaces.Box(low=limits[0], high=limits[1], shape=(1,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # below is effectively nothing
        self.context_inputs = spaces.Box(low=-1.5, high=1.5, shape=(0,), dtype=np.float32)
        # self.sum = 0
        self.state = np.zeros(1)
        self.n_timesteps = n_timesteps
        self.noise = noise
        self.switch_prob = switch_prob
        # special just to plot the difference
        # self.inputs_diff = np.zeros(n_timesteps)
        self.loss_func = NBFFLoss(transition_blind=transition_blind)
        # remaining attributes
        self.input_labels = ["Input"]
        self.output_labels = ["Output"]
        self.coupled_env = False
        self.transition_blind = transition_blind
        self.n = n
        self.poisson = poisson
        
    def simulate_poisson_process_single_arrivals(self):
        # Calculate the rate parameter alpha
        alpha = self.n / self.n_timesteps

        # Generate inter-arrival times
        inter_arrival_times = np.random.exponential(1/alpha, self.n)
        inter_arrival_times_cumulative = np.cumsum(inter_arrival_times)

        # Convert continuous time to discrete timesteps
        arrival_indices = np.floor(inter_arrival_times_cumulative).astype(int)

        # Create an array for the entire timestep period initialized to zero
        arrivals = np.zeros(self.n_timesteps, dtype=int)

        # Set the arrival indices to 1, ensuring no index is out of bounds or duplicated
        # Use np.unique to ensure no index is repeated
        filtered_indices = np.unique(arrival_indices[arrival_indices < self.n_timesteps])
        arrivals[filtered_indices] = 1

        return arrivals
        
        
    def step(self, action):
        self.state[0] = np.sign(action)
        
    def generate_trial(self):
        self.reset()
        
        # Generate the times when the bit flips
        inputRand = np.random.random(size=(self.n_timesteps,1))
        inputs_bits = np.zeros((self.n_timesteps, 1))
        
        if self.limits is not None:
            inputs_diff = np.zeros(self.n_timesteps)
            this_sum = 0
            # will always start at 0
            inputs_bits[0] = 0
            inputs_diff[0] = 0
            
            # get the time indices for sparse pulses
            if self.poisson:
                time_indices = self.simulate_poisson_process_single_arrivals()
            else:
                # Initialize an array of zeros
                time_indices = np.zeros(self.n_timesteps)

                # Select self.n random indices
                random_indices = np.random.choice(range(self.n_timesteps), self.n, replace=False)

                # Set the selected indices to 1
                time_indices[random_indices] = 1
                
            # send pulses while keeping the sum
            for i in range(2,self.n_timesteps):
                if this_sum == self.limits[1] and time_indices[i] == 1:
                    inputs_bits[i] = -1
                elif this_sum == self.limits[0] and time_indices[i] == 1:
                    inputs_bits[i] = 1
                elif time_indices[i] == 1:
                    inputs_bits[i] = np.random.choice([-1,1])
                this_sum += inputs_bits[i]
                inputs_diff[i] = this_sum
        else:
            inputs_bits[inputRand > (1 - self.switch_prob)] = 1  
            inputs_bits[inputRand < (self.switch_prob)] = -1
        
        # Generate DESIRED outputs given inputs
        outputs = np.zeros((self.n_timesteps,1))
        for i in range(self.n_timesteps):
            self.step(inputs_diff[i])
            outputs[i,:] = self.state
        
        # Add noise to the inputs for the trial
        noisy_inputs = inputs_bits + np.random.normal(loc=0.0, scale=self.noise, size=inputs_bits.shape)
        return noisy_inputs, outputs, inputs_bits, inputs_diff
        
    def reset(self):
        self.sum = 0
        self.state = np.zeros(1)
        return self.state
    
    def generate_dataset(self, n_samples):
        # Generates a dataset for the NBFF task
        # NOTE are we initializing always with zeros?
        ics_ds = np.zeros(shape=(n_samples, 1))
        outputs_ds = np.zeros(shape=(n_samples, self.n_timesteps, 1))
        inputs_ds = np.zeros(shape=(n_samples, self.n_timesteps, 1))
        true_inputs_ds = np.zeros(shape=(n_samples, self.n_timesteps, 1))
        for i in range(n_samples):
            inputs, outputs, true_inputs, _ = self.generate_trial()
            outputs_ds[i, :, :] = outputs
            inputs_ds[i, :, :] = inputs
            true_inputs_ds[i, :, :] = true_inputs

        dataset_dict = {
            "ics": ics_ds,
            "inputs": inputs_ds,
            "inputs_to_env": np.zeros(shape=(n_samples, self.n_timesteps, 0)),
            "targets": outputs_ds,
            "true_inputs": true_inputs_ds,
            "conds": np.zeros(shape=(n_samples,1)),
            # No extra info for this task, so just fill with zeros
            "extra": np.zeros(shape=(n_samples, 1)),
        }
        return dataset_dict
    
    def render(self):
        # Generate the trial data
        inputs, outputs, true_inputs, inputs_diff = self.generate_trial()

        # Create a figure
        fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True)

        # Plot true_inputs
        axs[0].plot(true_inputs[:,0], color='g')
        axs[0].set_ylabel("\n".join(textwrap.wrap("Noiseless Inputs", 12)))

        # Plot inputs
        axs[1].plot(inputs[:,0], color='r')
        axs[1].set_ylabel("\n".join(textwrap.wrap("Inputs", 12)))

        # Plot accumulated difference
        axs[2].plot(inputs_diff, color='k')
        axs[2].set_ylabel("\n".join(textwrap.wrap("Accumulated input sum", 12)))

        # Plot outputs
        axs[3].plot(outputs[:,0], color='b')
        axs[3].set_ylabel("\n".join(textwrap.wrap("Targets", 12)))

        # Set the x-label for the last subplot
        axs[3].set_xlabel("Time")

        plt.figure(dpi=500)
        # Display the plot
        plt.show()

        plt.figure(dpi=500)
        # Display the
