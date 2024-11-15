import random
from sklearn.linear_model import LogisticRegression
import numpy as np

# 0 = split, 1 = steal
MAX_SCORE = 5
SPLIT_SCORE = 3
DISAGREE_SCORE = 0
BOTH_STEAL_SCORE = 1
MAJOR_ITERATIONS = 100

# Helper Methods
def evaluate_choices(c1, c2):
    if c1 == 0 and c2 == 0:  # Both split
        return SPLIT_SCORE, SPLIT_SCORE
    elif c1 == 1 and c2 == 0:  # Agent 1 steals, Agent 2 splits
        return MAX_SCORE, DISAGREE_SCORE
    elif c1 == 0 and c2 == 1:  # Agent 1 splits, Agent 2 steals
        return DISAGREE_SCORE, MAX_SCORE
    else:  # Both steal
        return BOTH_STEAL_SCORE, BOTH_STEAL_SCORE

class Agent:
    def __init__(self, name="Agent"):
        self.name = name
        self.score = 0
        self.wins = 0
        self.conquered = []
        self.own_history = []
        self.opponent_history = []

    def choose(self):
        """Base method to make a choice in the prisoner's dilemma game.
        Should be overridden by child classes."""
        raise NotImplementedError("Subclasses should implement this method")

    def update_score(self, score):
        """Updates the agent's score by adding the provided score value."""
        self.score += score

    def record_move(self, choice, opponent_choice):
        """Records the agent's own choice and the opponent's last choice."""
        self.own_history.append(choice)
        self.opponent_history.append(opponent_choice)

    def clear_memory(self):
        self.opponent_history = []
        self.own_history = []

class UserAgent(Agent):
    def __init__(self):
        super().__init__(name="User Agent")

    def choose(self):
        """Prompts the user to choose between split (0) or steal (1)."""
        while True:
            try:
                choice = int(input("Do you want to split (0) or steal (1)? Enter number for choice: "))
                if choice in [0, 1]:
                    return choice
                else:
                    print("Invalid choice. Please enter 0 or 1.")
            except ValueError:
                print("Invalid input. Please enter a number (0 or 1).")

class RandomAgent(Agent):
    def __init__(self):
        super().__init__(name="Random Agent")

    def choose(self):
        """Randomly chooses between split (0) or steal (1)."""
        choice = random.choice([0, 1])
        return choice

class AlwaysSplitAgent(Agent):
    def __init__(self):
        super().__init__(name="Always Split Agent")

    def choose(self):
        """Always chooses to split (0)."""
        choice = 0
        return choice

class AlwaysStealAgent(Agent):
    def __init__(self):
        super().__init__(name="Always Steal Agent")

    def choose(self):
        """Always chooses to steal (1)."""
        choice = 1
        return choice

class TitForTatAgent(Agent):
    def __init__(self):
        super().__init__(name="Tit-for-Tat Agent")

    def choose(self):
        """Always chooses to cooperate on first turn, then mirrors opponent's previous move."""
        choice = 0 if not self.own_history else self.opponent_history[-1]
        return choice

class RhythmicAgent(Agent):
    def __init__(self, sequence_num=3, majority_split=True):
        super().__init__(name="Rhythmic Agent")
        self.sequence_num = sequence_num
        self.majority_split = majority_split
        self.count = 1

    def choose(self):
        """Splits or Steals every n-th iteration."""
        is_nth_iteration = self.count % self.sequence_num == 0
        choice = 1 if (self.majority_split and is_nth_iteration) else 0
        if not self.majority_split:
            choice = 0 if is_nth_iteration else 1
        self.count += 1
        return choice

class ProbabilisticAgent(Agent):
    def __init__(self, prob=0.75):
        super().__init__(name="Probabilistic Agent")
        self.probability = prob

    def choose(self):
        choice = 0 if random.random() < self.probability else 1
        return choice

class PredictionAgent(Agent):
    def __init__(self, pattern_length=2):
        super().__init__(name="Prediction Agent")
        self.pattern_length = pattern_length

    def choose(self):
        """Predicts the opponent's next move based on the most frequent pattern in history."""

        if len(self.opponent_history) >= self.pattern_length:
            choice = self.predict_next_move()
        else:
            choice = 0  # Default to "Split" if not enough history

        return choice

    def predict_next_move(self):
        """Predicts the opponent's next move based on historical patterns."""
        recent_pattern = tuple(self.opponent_history[-self.pattern_length:])
        pattern_counts = {}

        for i in range(len(self.opponent_history) - self.pattern_length):
            pattern = tuple(self.opponent_history[i:i + self.pattern_length])
            next_move = self.opponent_history[i + self.pattern_length] if i + self.pattern_length < len(self.opponent_history) else None

            if pattern == recent_pattern and next_move is not None:
                pattern_counts[next_move] = pattern_counts.get(next_move, 0) + 1

        if pattern_counts:
            return max(pattern_counts, key=pattern_counts.get)

        return 0  # Default to split if no pattern found

class GrudgeAgent(Agent):
    def __init__(self):
        super().__init__(name="Grudge Agent")
        self.grudge = False

    def choose(self):
        if self.opponent_history:
            if self.opponent_history[-1] == 1:
                self.grudge = True
        choice = 1 if self.grudge else 0
        return choice

    def clear_memory(self):
        super().clear_memory()
        self.grude = False
    
class MLPredictionAgent(Agent):
    def __init__(self):
        super().__init__(name="ML Prediction Agent")
        self.model = LogisticRegression()
        self.features = []
        self.labels = []

    def choose(self):
        """Predicts the opponent's next move using a machine learning model trained on historical data,
           including both the opponent's moves and the agent's own past guesses as features."""

        # Gather data if there are at least four moves to form features and labels
        if len(self.opponent_history) >= 4:
            # Last three moves of both opponent and agent as features
            feature = self.opponent_history[-4:-1] + self.own_history[-4:-1]
            self.features.append(feature)
            self.labels.append(self.opponent_history[-1])  # Next opponent action as the label

            # Convert lists to numpy arrays for training
            X = np.array(self.features)
            y = np.array(self.labels)

            # Only retrain if we have at least two different classes
            if len(set(y)) > 1:
                self.model.fit(X, y)

                # Make a prediction based on the latest sequence of moves
                prediction = self.model.predict([self.opponent_history[-3:] + self.own_history[-3:]])[0]
                self.own_history.append(prediction)  # Track this guess for the next round
                 #if the opponent is going to split, we steal
                if prediction == 0:
                    return 1
                # if they steal, we are going to split to maximize our score
                else:
                    return 0
            else:
                # Not enough class diversity, default to "split"
                self.own_history.append(0)
                return random.choice([0, 1])
        else:
            # Not enough data yet, so split by default
            self.own_history.append(0)
            return random.choice([0, 1])

    def clear_memory(self):
        super().clear_memory()
        self.model = LogisticRegression()
        self.features = []
        self.labels = []


class QLearningAgent(Agent):
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.2, history_states=35):
        super().__init__(name="Q-Learning Agent")
        self.q_table = {}  # Q-table initialized as a dictionary for state-action pairs
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.exploration_rate = exploration_rate  # Epsilon
        self.last_state = None
        self.last_action = None
        self.history_states = history_states

    def get_state(self, history_states):
        """Encodes the last (history_states) opponent moves as the current state."""
        if len(self.opponent_history) >= history_states:
            return tuple(self.opponent_history[-history_states:])
        else:
            # If there are fewer than history_states moves, pad with splits (0)
            return tuple([0] * (history_states - len(self.opponent_history)) + self.opponent_history)

    def choose(self):
        """Chooses an action based on the Q-learning policy (epsilon-greedy)."""
        state = self.get_state(self.history_states)

        # Initialize Q-values for this state if not present
        if state not in self.q_table:
            self.q_table[state] = [0, 0]  # Initialize Q-values for actions: [split, steal]
        # Epsilon-greedy action selection
        if random.random() < self.exploration_rate:
            action = random.choice([0, 1])  # Explore: choose random action
        else:
            action = 0 if self.q_table[state][0] >= self.q_table[state][1] else 1  # Exploit: choose best action

        # Save state and action for learning update
        self.last_state = state
        self.last_action = action

        return action

    def record_move(self, choice, opponent_choice):
        """Records the agent's own choice and the opponent's last choice,
           and updates Q-values immediately."""
        super().record_move(choice, opponent_choice)
        reward, _ = evaluate_choices(choice, opponent_choice)  # Get reward for the action

        if self.last_state is not None and self.last_action is not None:
            # Get the maximum Q-value for the next state
            next_state = self.get_state(self.history_states)
            if next_state not in self.q_table:
                self.q_table[next_state] = [0, 0]  # Initialize Q-values for actions: [split, steal]
            max_next_q = max(self.q_table[next_state])

            # Update Q-value using the Q-learning formula
            old_value = self.q_table[self.last_state][self.last_action]
            self.q_table[self.last_state][self.last_action] = (
                old_value + self.learning_rate * (reward + self.discount_factor * max_next_q - old_value)
            )

        # Update last state and action
        self.last_state = self.get_state(self.history_states)
        self.last_action = choice

    def clear_memory(self):
        super().clear_memory()
        self.q_table = {}
        self.last_action = None
        self.last_state = None