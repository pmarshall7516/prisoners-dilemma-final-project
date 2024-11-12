import random

# 0 = split, 1 = steal

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
