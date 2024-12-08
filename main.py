#! /usr/bin/env python

import sys
import json
import time
import pandas as pd
from agent import (evaluate_choices, MAJOR_ITERATIONS, Agent, UserAgent, RandomAgent, AlwaysSplitAgent, 
                AlwaysStealAgent, TitForTatAgent, RhythmicAgent , ProbabilisticAgent,
                PredictionAgent, GrudgeAgent, MLPredictionAgent, QLearningAgent)

import matplotlib.pyplot as plt
import seaborn as sns
import os
    
def select_agent():
    """Displays a menu for selecting an agent and returns an instance of the chosen agent."""
    
    print("""
    Select an agent:
    1 - RandomAgent (randomly chooses between split and steal)
    2 - AlwaysSplitAgent (always chooses to split)
    3 - AlwaysStealAgent (always chooses to steal)
    4 - TitForTatAgent (uses the Tit-for-Tat strategy)
    5 - RhythmicAgent (splits or steals using a pattern)
    6 - ProbablisticAgent (splits based on a probability)
    7 - PredictionAgent (uses opponent pattern history to make decisions)
    8 - GrudgeAgent (splits until opponent deffects, then deffects every time)
    9 - ML Prediction Agent (more advanced prediction agent using Logit)
    10 - Q-Learning Agent (uses Q-Learning to make predictions and take action)
    """)
    
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if choice == 1:
                return RandomAgent()
            elif choice == 2:
                return AlwaysSplitAgent()
            elif choice == 3:
                return AlwaysStealAgent()
            elif choice == 4:
                return TitForTatAgent()
            elif choice == 5:
                return RhythmicAgent()
            elif choice == 6:
                return ProbabilisticAgent()
            elif choice == 7:
                return PredictionAgent()
            elif choice == 8:
                return GrudgeAgent()
            elif choice == 9:
                return MLPredictionAgent()
            elif choice == 10:
                return QLearningAgent()
            else:
                print("Invalid choice. Please enter a number 1-5.")
        except ValueError:
            print("Invalid input. Please enter a number 1-5.")

def user_game(iterations=10):
    """Simulates a prisoner's dilemma game for a set number of iterations 
    between a user-controlled agent and an opponent agent."""
    
    # Initialize the agents
    user_agent = UserAgent()
    opponent = select_agent()
    
    # Game loop
    for i in range(iterations):
        print(f"\n--- Round {i + 1} ---")
        
        # Get choices from both agents
        user_choice = user_agent.choose()
        opponent_choice = opponent.choose()

        user_agent.record_move(user_choice, opponent_choice)
        opponent.record_move(opponent_choice, user_choice)
        
        print(f"User chose: {'Split' if user_choice == 0 else 'Steal'}")
        print(f"Opponent chose: {'Split' if opponent_choice == 0 else 'Steal'}")

        user_score, opponent_score = evaluate_choices(user_choice, opponent_choice)

        user_agent.update_score(user_score)
        opponent.update_score(opponent_score)

        print(f"User's score: {user_agent.score}")
        print(f"Opponent's score: {opponent.score}")

    # Final scores
    print("\n--- Game Over ---")
    print(f"Final User Score: {user_agent.score}")
    print(f"Final Opponent Score: {opponent.score}\n")


def parse_simulation_config(json_path):
    """
    Parses a JSON file and returns a dictionary containing:
      - A dictionary of agents where the keys are agent types and the values are their hyperparameters.
      - The reward values from the simulation configuration.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        dict: Contains two keys:
              - 'agents': Dictionary of agent types and their hyperparameters.
              - 'rewards': Dictionary of reward values.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Extract agent information
    agents_with_hyperparameters = {}

    for agent in data.get("agents", []):
        agent_type = agent.get("type")
        hyperparameters = agent.get("hyperparameters", {})
        
        # Store the agent type as key and hyperparameters as value
        agents_with_hyperparameters[agent_type] = hyperparameters
    
    # Extract reward values
    rewards = data.get("simulation", {}).get("scores", {})
    itter = data.get("simulation", {}).get("max_iterations", {})

    return {
        "agents": agents_with_hyperparameters,
        "rewards": rewards,
        "iterations": itter
    }

    
def major_simulation(iterations=MAJOR_ITERATIONS, all=True, parameter_file = "parameter_configs/balanced.json"):
    """
    Conducts a large-scale round robin simulation with multiple agents and records their performance.
    Args:
        iterations (int): Number of iterations for each agent pair. Defaults to MAJOR_ITERATIONS.
        all (bool): Whether to include all agents in the simulation. Defaults to True.
        parameter_file (str): Path to the parameter configuration JSON file. Defaults to "parameter_configs/balanced.json".
    """

    # Parse hyperparameters from the JSON file
    hyperparameters = parse_simulation_config(parameter_file)
    agent_params = hyperparameters.get("agents", {})
    rewards = hyperparameters.get("rewards", {})
    iterations = hyperparameters.get("iterations")

    

    # Initialize hyper parameters
    # Rewards
    split_score = rewards["split_score"]
    disagree_score = rewards["disagree_score"]
    steal_score = rewards["both_steal_score"]
    max_score = rewards["max_score"]

    # Agents
    prob_agent_params = agent_params["ProbabilisticAgent"]
    pred_agent_params = agent_params["PredictionAgent"]
    rhythm_agent_params = agent_params["RhythmicAgent"]
    ml_agent_params = agent_params["MLPredictionAgent"]
    q_agent_params = agent_params["QLearningAgent"]



    # Instantiate each agent type
    if all: # Use all agents, including always split, always steal, and random
        agents = [
            RandomAgent(),
            AlwaysSplitAgent(),
            AlwaysStealAgent(),
            TitForTatAgent(),
            RhythmicAgent(sequence_num=rhythm_agent_params["sequence_num"], majority_split=rhythm_agent_params["majority_split"]),
            ProbabilisticAgent(prob= prob_agent_params["probability"]),
            PredictionAgent(pattern_length=pred_agent_params["pattern_length"]),
            GrudgeAgent(),
            MLPredictionAgent(history_states=ml_agent_params["history_states"]),
            QLearningAgent(learning_rate = q_agent_params["learning_rate"], discount_factor = q_agent_params["discount_factor"], 
                           exploration_rate= q_agent_params["exploration_rate"], history_states= q_agent_params["history_states"],
                           split = split_score, disagree =disagree_score, steal = steal_score, max_score = max_score)
        ]
    else: # Only Algorithmic and Complex agents
        agents = [
            TitForTatAgent(),
            RhythmicAgent(sequence_num=rhythm_agent_params["sequence_num"], majority_split=rhythm_agent_params["majority_split"]),
            ProbabilisticAgent(prob= prob_agent_params["probability"]),
            PredictionAgent(pattern_length=pred_agent_params["pattern_length"]),
            GrudgeAgent(),
            MLPredictionAgent(history_states=ml_agent_params["history_states"]),
            QLearningAgent(learning_rate = q_agent_params["learning_rate"], discount_factor = q_agent_params["discount_factor"], 
                           exploration_rate= q_agent_params["exploration_rate"], history_states= q_agent_params["history_states"],
                           split = split_score, disagree =disagree_score, steal = steal_score, max_score = max_score)
        ]

    print(f"Running Simulation for {iterations} Iterations...")

    highest_combined_score = 0
    top_pair = (None, None)

    agent_metrics = []
    pairwise_metrics = []
    
    for a1 in agents:  # Main agent
        opponent_score_differences = []
        best_partner_score = 0
        best_partner = None
        a1_cooperations = 0
        total_a1_rounds = 0

        for a2 in agents:  # Opposing agent (including self)
            opponent_round_score = 0
            main_round_score = 0

            if isinstance(a1, Agent) and isinstance(a2, Agent):
                # Reset previous actions for each pairing
                a1.clear_memory() 
                a2.clear_memory()

                for i in range(iterations):
                    # Run a simulation step for the pair of agents
                    c1 = a1.choose()
                    c2 = a2.choose()
                    # print(f"{a1.name}: {c1} against {a2.name}: {c2}")

                    if c1 == 0:
                        a1_cooperations += 1

                    # Update both agent's memory
                    a1.record_move(c1, c2)
                    a2.record_move(c2, c1)

                    s1, s2 = evaluate_choices(c1, c2, split_score, disagree_score, steal_score, max_score)
                    # print(f"{a1.name}: {s1}, {a2.name}: {s2}")

                    # Calculate round by round score
                    main_round_score += s1
                    opponent_round_score += s2
                    total_a1_rounds += 1
                    
                
                # Update scores and metrics after each matchup
                a1.score += main_round_score
                score_difference = main_round_score - opponent_round_score
                opponent_score_differences.append(score_difference)

                # Calculate the combined score for the pairing
                combined_score = main_round_score + opponent_round_score
                a1.collective_score += combined_score
                if combined_score > highest_combined_score:
                    highest_combined_score = combined_score
                    top_pair = (a1, a2)

                # Update win count if this agent outscored the opponent
                if main_round_score > opponent_round_score:
                    a1.wins += 1

                if combined_score > best_partner_score:
                    best_partner_score = combined_score
                    best_partner = a2.name

                # Record pairwise data
                pairwise_metrics.append({
                    "Agent1": a1.name,
                    "Agent2": a2.name,
                    "Agent1_Score": main_round_score,
                    "Agent2_Score": opponent_round_score,
                    "Combined_Score": combined_score
                })


        # Calculate average metrics for the agent
        average_score = a1.score / len(agents)  # Include self-match in the average
        win_percentage = (a1.wins / len(agents)) * 100
        average_score_difference = sum(opponent_score_differences) / len(opponent_score_differences)

        agent_metrics.append({
            "Agent": a1.name,
            "Average_Score": average_score,
            "Win_Percentage": win_percentage,
            "Average_Score_Difference": average_score_difference,
            "Best_Partner": best_partner,
            "Best_Collective_Score": best_partner_score,
            "Average_Collective_Score": a1.collective_score / total_a1_rounds,
            "Cooperation_Rate": a1_cooperations / total_a1_rounds
        })
        
        # print(f"{a1.name} - Average Score: {average_score:.2f}, "
        #       f"Win Percentage: {win_percentage:.2f}%, "
        #       f"Average Opponent Score Difference: {average_score_difference:.2f}")
        # print(f"    Best Partner for {a1.name}: {best_partner} with a combined score of {best_partner_score}")

    
    # Create DataFrames
    agent_df = pd.DataFrame(agent_metrics)
    pairwise_df = pd.DataFrame(pairwise_metrics)

    # Save or display the results
    agent_df.to_csv(f"sim_logs/agent_metrics_{'all' if all else 'subset'}.csv", index=False)
    pairwise_df.to_csv(f"sim_logs/pairwise_metrics_{'all' if all else 'subset'}.csv", index=False)

    # print("Agent Metrics:")
    # print(agent_df)
    # print("\nPairwise Metrics:")
    # print(pairwise_df)

    # Determine the top agents based on each performance metric
    top_average_score_agent = agent_df.loc[agent_df['Average_Score'].idxmax()]
    top_win_rate_agent = agent_df.loc[agent_df['Win_Percentage'].idxmax()]
    bottom_average_score_agent = agent_df.loc[agent_df['Average_Score'].idxmin()]

    # Simulation Results sim print output
    print("\nResults Summary:")
    print(f"Top Agent by Average Score: {top_average_score_agent['Agent']} - Score: {top_average_score_agent['Average_Score']:.2f}")
    print(f"Top Agent by Win Percentage: {top_win_rate_agent['Agent']} - Win Percentage: {top_win_rate_agent['Win_Percentage']:.2f}%")

    if top_pair[0] and top_pair[1]:
        print(f"Highest Combined Score Pair: {top_pair[0].name} and {top_pair[1].name} - Combined Score: {highest_combined_score}")

    print(f"Worst Agent by Average Score: {bottom_average_score_agent['Agent']} - Score: {bottom_average_score_agent['Average_Score']:.2f}")

    # Plot the metrics
    plot_agent_metrics(agent_df)


def minor_simulation(iterations=10):
    """
    Conducts a smaller simulation between two user-selected agents and shows step-by-step results.
    Args:
        iterations (int): Number of rounds to simulate. Defaults to 10.
    """

    print("First Agent Selection...")
    a1 = select_agent()
    time.sleep(0.5)
    print("Second Agent Selection...")
    a2 = select_agent()
    time.sleep(0.5)
    
    if isinstance(a1, Agent) and isinstance(a2, Agent):
        # Game loop
        for i in range(iterations):
            print(f"\n--- Round {i + 1} ---")
            
            # Get choices from both agents
            c1 = a1.choose()
            c2 = a2.choose()

            # Update both agent's memory
            a1.record_move(c1, c2)
            a2.record_move(c2, c1)

            time.sleep(1)
            print(f"{a1.name} chose: {'Split' if c1 == 0 else 'Steal'}")
            print(f"{a2.name} chose: {'Split' if c2 == 0 else 'Steal'}")

            # Calculate rewards based on choices
            s1, s2 = evaluate_choices(c1, c2)

            a1.score += s1
            a2.score += s2
            time.sleep(0.5)

            print(f"{a1.name}'s score: {a1.score}")
            print(f"{a2.name}'s score: {a2.score}")
            time.sleep(1.5)

    # Final scores
    print("\n--- Game Over ---")
    print(f"Final {a1.name}'s Score: {a1.score}")
    print(f"Final {a2.name}'s Score: {a2.score}\n")

def main():
    """
    Entry point for the simulation. Handles user input for different simulation modes.

        -u: starts a game with user agent against selected agent
        -s: runs major round robin simulation (just algorithmic and complex agents)
            -a: directly follows -s flag and runs with simple agents as well
            * follow all flags with parameter file name (no parameter_configs/) to use specific parameter setups
            * EX. python main.py -s -a balanced.json
        -t: runs minor simulation which prompts the user to selects 2 agents and the number of iterations
            and watch the simulation of the iterated prisoner's dilemma game            
    """
    if len(sys.argv) < 2:
        major_simulation(MAJOR_ITERATIONS, True)
    else: 
        mode = sys.argv[1]
        
        if mode == "-u":
            try:
                iters = input("How many iterations would you like this game to last? (Default 10): ")
                user_game(int(iters) if iters else 10)
            except ValueError:
                print("Invalid input. Using default of 10 iterations.")
                user_game()
        elif mode == "-s":
            if len(sys.argv) > 2:
                param_file = "parameter_configs/"+sys.argv[-1] if sys.argv[-1] != None else "parameter_configs/balanced.json"
                if sys.argv[2] == '-a':
                    major_simulation(MAJOR_ITERATIONS, True, parameter_file=param_file)
                else:
                    major_simulation(MAJOR_ITERATIONS, False, parameter_file=param_file)
            else:
                major_simulation(MAJOR_ITERATIONS, False, parameter_file=param_file)
        elif mode == "-t":
            try:
                iters = input("How many iterations would you like this game to last? (Default 10): ")
                minor_simulation(int(iters) if iters else 10)
            except ValueError:
                print("Invalid input. Using default of 10 iterations.")
                minor_simulation()



def plot_agent_metrics(df):
    """
    Plots agent metrics as bar charts and saves the visualizations.
    Args:
        df (pd.DataFrame): DataFrame containing agent metrics.
    """
    metrics_to_plot = ['Average_Score', 'Win_Percentage', 'Average_Score_Difference', 
                       'Best_Collective_Score', 'Average_Collective_Score', 'Cooperation_Rate']

    # Set the plot style
    sns.set_theme(style="whitegrid")

    # Create a bar chart for each metric
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        # Use the x variable as hue for coloring
        sns.barplot(x='Agent', y=metric, data=df, hue='Agent', palette="viridis", dodge=False)
        
        # Suppress the legend as 'hue' is used only for coloring
        plt.legend([], [], frameon=False)
        
        plt.title(f"Comparison of Agents by {metric.replace('_', ' ')}", fontsize=14)
        plt.ylabel(metric.replace('_', ' '), fontsize=12)
        plt.xlabel("Agent", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()

if __name__ == "__main__":
    main()