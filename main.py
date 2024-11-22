#! /usr/bin/env python

import sys
import time
import pandas as pd
from agent import (evaluate_choices, MAX_SCORE, SPLIT_SCORE, DISAGREE_SCORE, BOTH_STEAL_SCORE, MAJOR_ITERATIONS, Agent, UserAgent, RandomAgent, AlwaysSplitAgent, 
                AlwaysStealAgent, TitForTatAgent, RhythmicAgent , ProbabilisticAgent,
                PredictionAgent, GrudgeAgent, MLPredictionAgent, QLearningAgent)
    
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
    
def major_simulation(iterations=MAJOR_ITERATIONS, all=True):
    # Instantiate each agent type
    if all:
        agents = [
            RandomAgent(),
            AlwaysSplitAgent(),
            AlwaysStealAgent(),
            TitForTatAgent(),
            RhythmicAgent(sequence_num=3, majority_split=True),
            ProbabilisticAgent(prob=0.75),
            PredictionAgent(pattern_length=2),
            GrudgeAgent(),
            MLPredictionAgent(),
            QLearningAgent()
        ]
    else:
        agents = [
            TitForTatAgent(),
            RhythmicAgent(sequence_num=3, majority_split=True),
            ProbabilisticAgent(prob=0.75),
            PredictionAgent(pattern_length=2),
            GrudgeAgent(),
            MLPredictionAgent(),
            QLearningAgent()
        ]
        # agents = [
        #     AlwaysStealAgent(),
        #     TitForTatAgent(),
        #     GrudgeAgent(),
        # ]
        

    print(f"Running Simulation for {iterations} Iterations...")

    highest_combined_score = 0
    top_pair = (None, None)

    agent_metrics = []
    pairwise_metrics = []
    
    for a1 in agents:  # Main agent
        opponent_score_differences = []
        best_partner_score = 0
        best_partner = None

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

                    a1.record_move(c1, c2)
                    a2.record_move(c2, c1)

                    s1, s2 = evaluate_choices(c1, c2)
                    # print(f"{a1.name}: {s1}, {a2.name}: {s2}")

                    main_round_score += s1
                    opponent_round_score += s2
                
                # Update scores and metrics after each matchup
                a1.score += main_round_score
                score_difference = main_round_score - opponent_round_score
                opponent_score_differences.append(score_difference)

                # Calculate the combined score for the pairing
                combined_score = main_round_score + opponent_round_score
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
            "Best_Partner_Score": best_partner_score
        })
        
        # print(f"{a1.name} - Average Score: {average_score:.2f}, "
        #       f"Win Percentage: {win_percentage:.2f}%, "
        #       f"Average Opponent Score Difference: {average_score_difference:.2f}")
        # print(f"    Best Partner for {a1.name}: {best_partner} with a combined score of {best_partner_score}")

    
    # Create DataFrames
    agent_df = pd.DataFrame(agent_metrics)
    pairwise_df = pd.DataFrame(pairwise_metrics)

    # Save or display the results
    agent_df.to_csv(f"sim_logs/agent_metrics_{"all" if all else "subset"}.csv", index=False)
    pairwise_df.to_csv(f"sim_logs/pairwise_metrics_{"all" if all else "subset"}.csv", index=False)

    print("Agent Metrics:")
    print(agent_df)
    print("\nPairwise Metrics:")
    print(pairwise_df)

    # Determine the top agents based on each performance metric
    top_average_score_agent = agent_df.loc[agent_df['Average_Score'].idxmax()]
    top_win_rate_agent = agent_df.loc[agent_df['Win_Percentage'].idxmax()]
    bottom_average_score_agent = agent_df.loc[agent_df['Average_Score'].idxmin()]

    print("\nResults Summary:")
    print(f"Top Agent by Average Score: {top_average_score_agent['Agent']} - Score: {top_average_score_agent['Average_Score']:.2f}")
    print(f"Top Agent by Win Percentage: {top_win_rate_agent['Agent']} - Win Percentage: {top_win_rate_agent['Win_Percentage']:.2f}%")

    if top_pair[0] and top_pair[1]:
        print(f"Highest Combined Score Pair: {top_pair[0].name} and {top_pair[1].name} - Combined Score: {highest_combined_score}")

    print(f"Worst Agent by Average Score: {bottom_average_score_agent['Agent']} - Score: {bottom_average_score_agent['Average_Score']:.2f}")

def minor_simulation(iterations=10):
    """Simulates a prisoner's dilemma game for a set number 
    of iterations between two agents. Shows the steps"""

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

            a1.record_move(c1, c2)
            a2.record_move(c2, c1)

            time.sleep(1)
            print(f"{a1.name} chose: {'Split' if c1 == 0 else 'Steal'}")
            print(f"{a2.name} chose: {'Split' if c2 == 0 else 'Steal'}")

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
                if sys.argv[2] == '-a':
                    major_simulation(MAJOR_ITERATIONS, True)
                else:
                    major_simulation(MAJOR_ITERATIONS, False)
            else:
                major_simulation(MAJOR_ITERATIONS, False)
        elif mode == "-t":
            try:
                iters = input("How many iterations would you like this game to last? (Default 10): ")
                minor_simulation(int(iters) if iters else 10)
            except ValueError:
                print("Invalid input. Using default of 10 iterations.")
                minor_simulation()

if __name__ == "__main__":
    main()