# Prisoner's Dilemma Final Project
### Patrick Marshall and Christian Basso
##### CSC 4631 111

The Prisoner's Dilemma is a popular game theory scenario where two agents are pit against one another, and each are given two options: to cooperate or to defect. Based on their choices, the agents are rewarded with either negative, neutral, or positive score.

We implemented an iterative version of this game where each agent has a particular strategy, and plays multiple other agents in a round robin 400 round simulation. This simulation investigates multi-agent and non-zero-sum environments as well as helps us understand conditions that foster cooperation or competition. 

### File Descriptions

**main.py:** contains the main simulation loops for a major simulation (used to generate our data), a minor simulation (to visualize two particular agents play), and a user simulation so the user can play against our agents.

Also contains the data storage and plot visualization functions
utilizing numpy, pandas, and matplotlib

**agent.py:** contains all of the classes for our different agents. This includes all attributes and 
methods needed in order to implement their decision logic as well as resetting them for new game
loops. 

Also contains score evaluation method and default iteration value.

### Directory Descriptions

**parameter_configs/:** contains the *.json* files that are read by the program. These files contain the agent and environmental hyperparameters such as reward/iteration values and history_states.

**plots/:** after major simualtion is run, the resulting plots are saved here. They are all bar charts which depict the metrics and results of the experiment. 

**sim_logs/:** contains the *.csv* files of the numerical metric data after major simulation is run. Files containing *_subset are those where simple agents (always split/steal, random) are not used (as they skew results and are too predictable)

### How to Run the Code?

#### Prerequisites:
1. Install the required libraries:
```
pip install numpy pandas matplotlib
```

#### Running Simulations:
The program accepts different modes of operation, controlled via command-line arguments:

1. **User Simulation** (`-u`):
- Allows a user to play against a selected agent.
- Command to run:
  ```
  python main.py -u
  ```
- You will be prompted to input the number of iterations for the game (default is 10 if no input is provided).

2. **Major Simulation** (`-s`):
- Runs a round-robin tournament among all agents for multiple iterations.
- Optional flag `-a` includes simple agents (e.g., Always Split/Always Steal).
- Command to run:
  ```
  python main.py -s [optional -a] [parameter_file.json]
  ```
- Example:
  ```
  python main.py -s -a balanced.json
  ```
- If no parameter file is specified, the default `parameter_configs/balanced.json` is used.

3. **Minor Simulation** (`-t`):
- Allows selection of two specific agents to simulate a smaller number of games.
- Command to run:
  ```
  python main.py -t
  ```
- You will be prompted to select the agents.
- You will be prompted to input the number of iterations (default is 10 if no input is provided).

#### Notes:
- Replace `[parameter_file.json]` with the name of a valid configuration file from the `parameter_configs/` directory.
- Outputs from simulations are saved in the following directories:
- **`plots/`**: Visualizations (bar charts) from major simulations.
- **`sim_logs/`**: CSV logs of numerical data from simulations.



