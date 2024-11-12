import random
from sklearn.linear_model import LogisticRegression
import numpy as np

# 0 = split, 1 = steal

class Agent:
    def __init__(self, name="Agent"):
        self.name = name
        self.score = 0
        self.wins = 0
        self.previous_choice = None
        self.conquered = []

    def choose(self, opponent_last_choice):
        """Base method to make a choice in the prisoner's dilemma game.
        Should be overridden by child classes."""
        raise NotImplementedError("Subclasses should implement this method")
    
    def update_score(self, score):
        """Updates the agent's score by adding the provided score value."""
        self.score += score

    def get_previous_action(self):
        return self.previous_choice
    
    def set_previous_action(self, pr):
        self.previous_choice = pr
    
class UserAgent(Agent):
    def __init__(self):
        super().__init__()

    def choose(self, opponent_last_choice):
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

    def choose(self, opponent_last_choice):
        """Randomly chooses between split (0) or steal (1)."""
        choice = random.choice([0, 1])
        return choice
    
class AlwaysSplitAgent(Agent):
    def __init__(self):
        super().__init__(name="Always Split Agent")

    def choose(self, opponent_last_choice):
        """Always chooses to split (0)."""
        return 0
    
class AlwaysStealAgent(Agent):
    def __init__(self):
        super().__init__(name="Always Steal Agent")

    def choose(self, opponent_last_choice):
        """Always chooses to steal (1)."""
        return 1
    
class TitForTatAgent(Agent):
    def __init__(self):
        self.count = 0
        super().__init__(name="Tit-for-Tat Agent")

    def choose(self, opponent_last_choice):
        """Always chooses to cooperate on first turn, 
        then replicates opponents previous move"""
        if self.count == 0: 
            self.count += 1
            return 0
        else: 
            return opponent_last_choice
        
class RhythmicAgent(Agent):
    def __init__(self, sequence_num=3, majority_split=True):
        self.majority_split = majority_split
        self.sequence_num = sequence_num
        self.count = 1
        super().__init__(name="Rhythmic Agent")

    def choose(self, opponent_last_choice):
        """Splits or Steals every n-th iteration"""
        is_nth_iteration = self.count % self.sequence_num == 0
        self.count += 1

        if self.majority_split:
            # Split most of the time, steal on every n-th iteration
            return 1 if is_nth_iteration else 0
        else:
            # Steal most of the time, split on every n-th iteration
            return 0 if is_nth_iteration else 1
        
class ProbabilisticAgent(Agent):
    def __init__(self, prob=0.75):
        self.probability = prob
        super().__init__(name="Probabilistic Agent")

    def choose(self, opponent_last_choice):
        return 0 if random.random() < self.probability else 1
    
class PredictionAgent(Agent):
    def __init__(self, pattern_length=2):
        self.history = []
        self.pattern_length = pattern_length
        super().__init__(name="Prediction Agent")

    def choose(self, opponent_last_choice):
        """Predicts the opponent's next move based on the most frequent pattern in history."""
        if opponent_last_choice is not None:
            self.history.append(opponent_last_choice)
        
        # Predict the next move based on pattern
        if len(self.history) >= self.pattern_length:
            predicted_move = self.predict_next_move()
        else:
            predicted_move = 0  # Default to "Split" if not enough history

        return predicted_move

    def predict_next_move(self):
        """Predicts the opponent's next move based on historical patterns."""
        # Extract the most recent pattern
        recent_pattern = tuple(self.history[-self.pattern_length:])
        pattern_counts = {}

        # Count occurrences of each pattern in history
        for i in range(len(self.history) - self.pattern_length):
            pattern = tuple(self.history[i:i + self.pattern_length])
            next_move = self.history[i + self.pattern_length] if i + self.pattern_length < len(self.history) else None

            if pattern == recent_pattern and next_move is not None:
                if next_move not in pattern_counts:
                    pattern_counts[next_move] = 0
                pattern_counts[next_move] += 1

        # If there is a pattern, choose the most frequent next move
        if pattern_counts:
            predicted_move = max(pattern_counts, key=pattern_counts.get)
            #print(f"Next Move: {predicted_move} based on pattern {recent_pattern}")
            return predicted_move

        # If no pattern is found, default to "Split" (0)
        return 0




class MLPredictionAgent(Agent):
    def __init__(self):
        super().__init__(name="ML Prediction Agent")
        self.history = []  # Tracks opponent's past choices
        self.guess_history = []  # Tracks the agent's own past guesses
        self.model = LogisticRegression()
        self.features = []
        self.labels = []

    def choose(self, opponent_last_choice):
        """Predicts the opponent's next move using a machine learning model trained on historical data,
           including both the opponent's moves and the agent's own past guesses as features."""
        
        # Append the opponent's and agent's last actions to history
        if opponent_last_choice is not None:
            self.history.append(opponent_last_choice)
        
        if len(self.guess_history) < len(self.history):
            # Append a default action to the guess history if it's shorter than the opponent's history
            self.guess_history.append(0)  # Default to split initially

        # Gather data if there are at least four moves to form features and labels
        if len(self.history) >= 4:
            # Last three moves of both opponent and agent as features
            feature = self.history[-4:-1] + self.guess_history[-4:-1]
            self.features.append(feature)
            self.labels.append(self.history[-1])  # Next opponent action as the label

            # Convert lists to numpy arrays for training
            X = np.array(self.features)
            y = np.array(self.labels)

            # Only retrain if we have at least two different classes
            if len(set(y)) > 1:
                self.model.fit(X, y)

                # Make a prediction based on the latest sequence of moves
                prediction = self.model.predict([self.history[-3:] + self.guess_history[-3:]])[0]
                self.guess_history.append(prediction)  # Track this guess for the next round
                 #if the opponent is going to split, we steal
                if prediction == 0:
                    return 1
                # if they steal, we are going to split to maximize our score
                else:
                    return 0
            else:
                # Not enough class diversity, default to "split"
                self.guess_history.append(0)
                return random.choice([0, 1])
        else:
            # Not enough data yet, so split by default
            self.guess_history.append(0)
            return random.choice([0, 1])
    