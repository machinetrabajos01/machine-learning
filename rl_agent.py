"""
Reinforcement Learning module (Activity R2A2).

10 x 10 grid world solved with Q-Learning. Q(s, a) is approximated with
scikit-learn's SGDRegressor: predict() estimates the Q-values and
partial_fit() updates the model after every observed transition.
"""

import numpy as np
from sklearn import config_context
from sklearn.linear_model import SGDRegressor


# Environment

# A = agent start, T = target, o = available path,
# # = wall / obstacle, D = danger zone
GRID_LAYOUT = [
    "Aoo#oooooo",
    "o#o#oDo##o",
    "o#ooo#ooDo",
    "ooD#o#o#oo",
    "#ooooooooD",
    "oo#oDo#ooo",
    "oDoooo#oDo",
    "o##o#Dooo#",
    "ooDo#oo#oo",
    "oooooDoooT",
]

GRID = [list(row) for row in GRID_LAYOUT]
ROWS = len(GRID)
COLS = len(GRID[0])

EXPECTED_COUNTS = {"A": 1, "T": 1, "o": 68, "#": 20, "D": 10}


def count_cells():
    counts = {symbol: 0 for symbol in EXPECTED_COUNTS}
    for row in GRID:
        for cell in row:
            counts[cell] += 1
    return counts


def find_cell(symbol):
    for r in range(ROWS):
        for c in range(COLS):
            if GRID[r][c] == symbol:
                return (r, c)
    raise ValueError(f"Symbol {symbol} not found in the grid")


assert count_cells() == EXPECTED_COUNTS, count_cells()

START = find_cell("A")
GOAL = find_cell("T")

ACTIONS = ["Up", "Down", "Left", "Right"]
ACTION_MOVES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}
N_ACTIONS = len(ACTIONS)

CELL_TYPES = {
    "A": "Start",
    "T": "Target",
    "o": "Path",
    "#": "Wall",
    "D": "Danger Zone",
}


# Reward system

REWARDS = {
    "step": -1,       # moving to a valid normal position
    "invalid": -3,    # trying to leave the grid
    "wall": -5,       # hitting a wall
    "danger": -15,    # entering a Danger Zone
    "goal": 100,      # reaching the target
}

REWARD_TABLE = [
    {
        "event": "Moving to a valid normal position (o / A)",
        "value": REWARDS["step"],
        "reason": "Small cost per movement so that shorter routes "
                  "accumulate a higher return.",
    },
    {
        "event": "Attempting an invalid movement (outside the grid)",
        "value": REWARDS["invalid"],
        "reason": "The agent stays in the same cell and wastes a step.",
    },
    {
        "event": "Hitting a wall (#)",
        "value": REWARDS["wall"],
        "reason": "The agent stays in the same cell; walls cost more "
                  "than a simple invalid move.",
    },
    {
        "event": "Entering a Danger Zone (D)",
        "value": REWARDS["danger"],
        "reason": "The move is allowed, but it is expensive enough that "
                  "a detour of several steps is still better.",
    },
    {
        "event": "Reaching the target (T)",
        "value": REWARDS["goal"],
        "reason": "Large positive reward; the episode ends.",
    },
]


# Training configuration

CONFIG = {
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.99,
    "episodes": 500,
    "max_steps": 150,
    "learning_rate": 0.2,
    "seed": 42,
}

CONFIG_TABLE = [
    {
        "name": "Initial epsilon (ε)",
        "value": CONFIG["epsilon_start"],
        "reason": "The agent knows nothing at first, so every action "
                  "is random (100% exploration).",
    },
    {
        "name": "Minimum epsilon",
        "value": CONFIG["epsilon_min"],
        "reason": "Keeps 5% of random actions so the agent never stops "
                  "checking alternative routes.",
    },
    {
        "name": "Epsilon decay",
        "value": CONFIG["epsilon_decay"],
        "reason": "After each episode ε = ε × 0.99, moving gradually "
                  "from exploration to exploitation.",
    },
    {
        "name": "Discount factor (γ)",
        "value": CONFIG["gamma"],
        "reason": "Future rewards matter almost as much as immediate "
                  "ones, so the far-away target can guide early steps.",
    },
    {
        "name": "Training episodes",
        "value": CONFIG["episodes"],
        "reason": "Enough attempts for ε to reach its minimum and for "
                  "the Q-values of a 10 × 10 grid to stabilise.",
    },
    {
        "name": "Maximum steps per episode",
        "value": CONFIG["max_steps"],
        "reason": "Stops episodes in which the agent wanders without "
                  "reaching the target.",
    },
    {
        "name": "Learning rate (SGDRegressor eta0)",
        "value": CONFIG["learning_rate"],
        "reason": "Size of each partial_fit() correction towards the "
                  "new Q-Learning target.",
    },
]


def step(state, action):
    """
    Apply one action to the environment.

    Returns: next_state, reward, done, cell_type
    """
    row, col = state
    d_row, d_col = ACTION_MOVES[action]
    new_row, new_col = row + d_row, col + d_col

    # Invalid movement: outside the grid
    if not (0 <= new_row < ROWS and 0 <= new_col < COLS):
        return state, REWARDS["invalid"], False, "Out of bounds"

    cell = GRID[new_row][new_col]

    # Walls cannot be crossed
    if cell == "#":
        return state, REWARDS["wall"], False, "Wall"

    next_state = (new_row, new_col)

    if cell == "T":
        return next_state, REWARDS["goal"], True, "Target"

    if cell == "D":
        return next_state, REWARDS["danger"], False, "Danger Zone"

    return next_state, REWARDS["step"], False, CELL_TYPES[cell]


# Q-value approximation with SGDRegressor

N_FEATURES = ROWS * COLS * N_ACTIONS


def feature_index(state, action):
    return (state[0] * COLS + state[1]) * N_ACTIONS + action


def encode(state, action):
    """One-hot encoding of the (state, action) pair."""
    features = np.zeros((1, N_FEATURES))
    features[0, feature_index(state, action)] = 1.0
    return features


def encode_all_actions(state):
    features = np.zeros((N_ACTIONS, N_FEATURES))
    base = feature_index(state, 0)
    for action in range(N_ACTIONS):
        features[action, base + action] = 1.0
    return features


def create_model(seed):
    model = SGDRegressor(
        learning_rate="constant",
        eta0=CONFIG["learning_rate"],
        penalty=None,
        fit_intercept=False,
        random_state=seed,
    )
    # First call creates the weights; all Q-values start at 0
    model.partial_fit(np.zeros((1, N_FEATURES)), [0.0])
    return model


def q_values(model, state):
    """Q(s, a) for the four actions, estimated with predict()."""
    return model.predict(encode_all_actions(state))


def choose_action(model, state, epsilon, rng):
    """Epsilon-Greedy action selection."""
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS)), "Exploration"
    return int(np.argmax(q_values(model, state))), "Exploitation"


# Training

def train_agent():
    rng = np.random.default_rng(CONFIG["seed"])
    model = create_model(CONFIG["seed"])

    gamma = CONFIG["gamma"]
    epsilon = CONFIG["epsilon_start"]

    episode_rewards = []
    successful_episodes = 0

    with config_context(assume_finite=True):

        for _ in range(CONFIG["episodes"]):
            state = START
            total_reward = 0

            for _ in range(CONFIG["max_steps"]):
                action, _ = choose_action(model, state, epsilon, rng)
                next_state, reward, done, _ = step(state, action)

                # Q-Learning target: r + γ · max Q(s', a')
                if done:
                    target = reward
                else:
                    target = reward + gamma * np.max(
                        q_values(model, next_state)
                    )

                model.partial_fit(encode(state, action), [target])

                total_reward += reward
                state = next_state

                if done:
                    successful_episodes += 1
                    break

            episode_rewards.append(total_reward)
            epsilon = max(
                CONFIG["epsilon_min"],
                epsilon * CONFIG["epsilon_decay"],
            )

        evaluation = evaluate_agent(model)
        q_table = build_q_table(model)

    episodes = CONFIG["episodes"]

    return {
        "episodes": episodes,
        "successful_episodes": successful_episodes,
        "success_rate": round(100 * successful_episodes / episodes, 2),
        "average_reward": round(float(np.mean(episode_rewards)), 2),
        "average_reward_last_100": round(
            float(np.mean(episode_rewards[-100:])), 2
        ),
        "final_epsilon": round(epsilon, 4),
        "evaluation": evaluation,
        "q_table": q_table,
    }


# Evaluation without exploration

def evaluate_agent(model):
    state = START
    path = [state]
    steps = []
    total_reward = 0
    reached_goal = False

    for number in range(1, CONFIG["max_steps"] + 1):
        action = int(np.argmax(q_values(model, state)))
        next_state, reward, done, cell_type = step(state, action)

        steps.append({
            "step": number,
            "state": state,
            "action": ACTIONS[action],
            "next_state": next_state,
            "cell_type": cell_type,
            "reward": reward,
        })

        total_reward += reward
        state = next_state
        path.append(state)

        if done:
            reached_goal = True
            break

    return {
        "steps": steps,
        "path": path,
        "movements": len(steps),
        "total_reward": total_reward,
        "reached_goal": reached_goal,
        "danger_entries": sum(
            1 for s in steps if s["cell_type"] == "Danger Zone"
        ),
        "collisions": sum(
            1 for s in steps
            if s["cell_type"] in ("Wall", "Out of bounds")
        ),
    }


def build_q_table(model):
    """Q-values of every valid (non-wall) state."""
    table = []

    for r in range(ROWS):
        for c in range(COLS):
            if GRID[r][c] == "#":
                continue

            values = q_values(model, (r, c))

            table.append({
                "state": (r, c),
                "cell": GRID[r][c],
                "values": [round(float(v), 2) for v in values],
                "best": ACTIONS[int(np.argmax(values))],
                "best_index": int(np.argmax(values)),
            })

    return table


def grid_view(path=None):
    """Grid prepared for the template, with the learned path marked."""
    order = {}
    for index, cell in enumerate(path or []):
        order.setdefault(cell, index)

    return [
        [
            {
                "symbol": GRID[r][c],
                "on_path": (r, c) in order,
                "order": order.get((r, c)),
            }
            for c in range(COLS)
        ]
        for r in range(ROWS)
    ]
