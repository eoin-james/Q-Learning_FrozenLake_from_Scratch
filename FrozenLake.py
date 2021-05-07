import numpy as np
import matplotlib
import pandas as pd

from matplotlib import pyplot as plt

# matplotlib.style.use('ggplot')

BoardSize = (5, 5)
WinPos = (4, 4)
StartPos = (0, 0)

TerminalReward = 10
HoleReward = -5
TransitionReward = -1

HolePos = (1, 0), (1, 3), (3, 1), (4, 2)

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
NumActions = 4

num_episodes = 10000
max_steps_per_episode = 1000

learning_rate = 0.1
gamma = 0.9
min_epsilon = 0.1
alpha = 0.5

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9996

q_table = np.zeros((BoardSize[0] * BoardSize[1], NumActions))

rewards_all_episodes = []


class Lake:
    def __init__(self,
                 state=StartPos,
                 end_pos=WinPos,
                 map_size=BoardSize,
                 holes=HolePos,
                 actions=(UP, DOWN, LEFT, RIGHT),
                 rewards=(TerminalReward, HoleReward, TransitionReward)):

        self.state = state
        self.done = False
        self.end_pos = end_pos

        self.nrow, self.ncol = map_size
        self.holes = holes
        self.actions = actions

        self.r_terminal, self.r_hole, self.r_transition = rewards

        self.ending_array = []

        self.map = self._gen_map()
        print('\n')
        print('Map')
        print(self.map)

    def _gen_map(self):
        """
        Generates the Frozen Lake
        :return: the map
        """
        mp = [['F' for _ in range(self.ncol)] for _ in range(self.nrow)]
        mp[self.state[0]][self.state[1]] = 'S'
        mp[self.end_pos[0]][self.end_pos[1]] = 'G'
        for h in self.holes:
            mp[h[0]][h[1]] = 'H'
        mp = ["".join(x) for x in mp]
        return np.asarray(mp, dtype='c')

    def get_reward(self):
        """
        Gets the reward returned for landing on the current state
        :return:
        """
        letter = self.map[self.state[0]][self.state[1]]
        reward = 0
        if letter in b'G':
            reward = self.r_terminal
        elif letter in b'H':
            reward = self.r_hole
        # Start is included as if start pos is by an edge, moving towards said edge
        # will result in landing in the same spot and this counts as a transition
        elif letter in b'FS':
            reward = self.r_transition
        return reward

    def move(self, act):
        """
        Moves the agent from the current position to a new position based on the action passed
        :param act: Which Action is being taken
        """
        row, col = self.state[0], self.state[1]

        if act == UP:
            row = max(row - 1, 0)
        elif act == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif act == LEFT:
            col = max(col - 1, 0)
        elif act == RIGHT:
            col = min(col + 1, self.ncol - 1)
        self.state = (row, col)

    def is_done(self):
        """
        Determines if the agent is done - Has reached a hole or a goal
        """
        self.done = bytes(self.map[self.state[0]][self.state[1]]) in b'GH'
        let = self.map[self.state[0]][self.state[1]]
        if self.done:
            self.ending_array.append([1 if let in b'G' else 0])
        if self.done:
            self.state = StartPos
        return self.done

    def step(self, action):
        self.move(action)  # Make agent take action and move to new state
        reward = self.get_reward()  # Get reward for taking that action
        done = self.is_done()  # Is game done
        new_state = self.state[0] * 5 + self.state[1]  # Get the new position of the agent and convert to Q Table lookup

        return reward, new_state, done


def game(q_tab, initial_eps):
    env = Lake()
    epsilon = initial_eps
    for episode in range(num_episodes):
        print('\n')
        print('Episode:', episode + 1, "/", num_episodes)
        episode_reward = 0
        state = env.state[0] * 5 + env.state[1]

        epsilon = epsilon * epsilon_decay

        for step in range(max_steps_per_episode):

            # Get Action based on Greedy Strategy
            eps = max(min_epsilon, epsilon)
            if np.random.random() <= eps:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(q_table[state, :])

            # Take Action to move to new state and get reward
            reward, new_state, done = env.step(action)
            episode_reward += reward

            # print(reward, new_state, done, action)

            # Update Q-Table
            q_tab[state, action] = q_tab[state, action] * (1 - learning_rate) + learning_rate * (
                    reward + gamma * np.max(q_tab[new_state, :]))

            state = new_state

            # If done go to next episode
            if done:
                print('Episode Reward:', episode_reward)
                print('Steps:', step + 1)
                rewards_all_episodes.append(episode_reward)
                break

    return q_tab, rewards_all_episodes, env.ending_array


def plot_result(rews, end):
    y = np.asarray(rews)

    x_wins = np.where(np.asarray(end) == 1)[0]
    x_fails = np.where(np.asarray(end) == 0)[0]

    y_wins = np.take(y, x_wins)
    y_fails = np.take(y, x_fails)

    plt.scatter(x=x_wins, y=y_wins, c='g', s=0.8, label='Win')
    plt.scatter(x=x_fails, y=y_fails, c='r', s=0.8, label='Fail')

    plt.text(1, 1, 'Wins')

    plt.title('Agent Reward Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    q_table, rewards, end = game(q_table, epsilon)
    print(len(end))

    df_q = pd.DataFrame(q_table)

    print('\n')
    print(df_q)

    print('\n')
    print('Wins:', end.count([1]))
    print('Fails:', end.count([0]))

    plot_result(rewards, end)
