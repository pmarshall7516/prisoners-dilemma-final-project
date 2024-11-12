#! /usr/bin/env python

import sys
import time
from agent import (Agent, UserAgent, RandomAgent, AlwaysSplitAgent, 
                AlwaysStealAgent, TitForTatAgent, RhythmicAgent , ProbabilisticAgent,
                PredictionAgent)

MAX_SCORE = 5
SPLIT_SCORE = 3
DISAGREE_SCORE = 0
BOTH_STEAL_SCORE = 1
MAJOR_ITERATIONS = 500

def evaluate_choices(c1, c2):
    if c1 == 0 and c2 == 0:  # Both split
        return SPLIT_SCORE, SPLIT_SCORE
    elif c1 == 1 and c2 == 0:  # Agent 1 steals, Agent 2 splits
        return MAX_SCORE, DISAGREE_SCORE
    elif c1 == 0 and c2 == 1:  # Agent 1 splits, Agent 2 steals
        return DISAGREE_SCORE, MAX_SCORE
    else:  # Both steal
        return BOTH_STEAL_SCORE, BOTH_STEAL_SCORE
    
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
    7 - PredictionAgent (Uses opponent pattern history to make decisions)
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
        user_choice = user_agent.choose(opponent.get_previous_action())
        opponent_choice = opponent.choose(user_agent.get_previous_action())

        user_agent.set_previous_action(user_choice)
        opponent.set_previous_action(opponent_choice)
        
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
    
def major_simulation(iterations=MAJOR_ITERATIONS):
    agents = [RandomAgent(), AlwaysSplitAgent(), AlwaysStealAgent(), 
              TitForTatAgent(), RhythmicAgent(), ProbabilisticAgent(),
              PredictionAgent()]
    
    print(f"Running Simulation for {iterations} Iterations...")
    for a1 in agents: # Main agent
        for a2 in agents: # Opposing agent
            opponent_round_score = 0
            main_round_score = 0
            if isinstance(a1, Agent) and isinstance(a2, Agent):
                for i in range(iterations):
                    # Run a simulation step for the pair of agents
                    c1 = a1.choose(a2.get_previous_action())
                    c2 = a2.choose(a1.get_previous_action())

                    a1.set_previous_action(c1)
                    a2.set_previous_action(c2)

                    s1, s2 = evaluate_choices(c1, c2)
                    
                    # Update the score for main agent
                    main_round_score += s1
                    opponent_round_score += s2
                
                a1.update_score(main_round_score)
                if main_round_score > opponent_round_score:
                    a1.wins += 1
                    a1.conquered.append(a2.name)

        print(f"{a1.name} - Average Score: {a1.score/len(agents)}, Win Percentage: {(a1.wins/len(agents))*100.0:.2f}")
        #print(f"    Conquered Agents: {a1.conquered}")
    
    # Determine agent with highest score and highest win percentage
    top_score_agent = max(agents, key=lambda agent: agent.score/len(agents))
    top_win_percentage_agent = max(agents, key=lambda agent: (agent.wins / len(agents)) * 100.0)

    print("\nResults Summary:")
    print(f"Agent with the highest average score: {top_score_agent.name} - Score: {top_score_agent.score/len(agents)}")
    print(f"Agent with the highest win percentage: {top_win_percentage_agent.name} - Win Percentage: {(top_win_percentage_agent.wins / len(agents)) * 100.0:.2f}")

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
            c1 = a1.choose(a2.get_previous_action())
            c2 = a2.choose(a1.get_previous_action())

            a1.set_previous_action(c1)
            a2.set_previous_action(c2)
            
            time.sleep(1)
            print(f"{a1.name} chose: {'Split' if c1 == 0 else 'Steal'}")
            print(f"{a2.name} chose: {'Split' if c2 == 0 else 'Steal'}")

            s1, s2 = evaluate_choices(c1, c2)

            a1.update_score(s1)
            a2.update_score(s2)
            time.sleep(0.5)

            print(f"{a1.name}'s score: {a1.score}")
            print(f"{a2.name}'s score: {a2.score}")
            time.sleep(1.5)

    # Final scores
    print("\n--- Game Over ---")
    print(f"Final {a1.name}'s Score: {a1.score}")
    print(f"Final {a2.name}'s Score: {a2.score}\n")

def main():
    mode = sys.argv[1]
    
    if mode == "-u":
        try:
            iters = input("How many iterations would you like this game to last? (Default 10): ")
            user_game(int(iters) if iters else 10)
        except ValueError:
            print("Invalid input. Using default of 10 iterations.")
            user_game()
    elif mode == "-s":
        major_simulation()
    elif mode == "-t":
        try:
            iters = input("How many iterations would you like this game to last? (Default 10): ")
            minor_simulation(int(iters) if iters else 10)
        except ValueError:
            print("Invalid input. Using default of 10 iterations.")
            minor_simulation()

if __name__ == "__main__":
    main()