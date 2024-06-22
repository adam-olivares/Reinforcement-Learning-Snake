import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import argparse
import csv
import os

class Agent:

    def __init__(self, discount, max_memory, learning_rate, batch_size):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = discount # discount rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=learning_rate, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        #print('State:', len(state))
        #print(state)
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(width, height, block_size, speed, discount, 
          max_memory, learning_rate, 
          batch_size, max_games, retrain=True, suffix=None):
    
    model_name = f'model_w={width}_h={height}_d={discount}_m={max_memory}_lr={learning_rate}_b={batch_size}'
        
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(discount, max_memory, learning_rate, batch_size)
    game = SnakeGameAI(w=width, h=height, block_size=block_size, speed=speed)

    max_dist = 9_999_999
    min_dist = None

    while True and agent.n_games < max_games: 
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        #print(final_move)

        # perform move and get new state
        # reward, game_over, self.score = play_step
        reward, done, score = game.play_step(final_move)
        old_reward = reward
        # minimize distance
        distance = np.abs(game.food.x - game.head.x) + np.abs(game.food.y - game.head.y)
        if min_dist is None: 
            min_dist = distance
        
        if distance > min_dist:
            reward -= 0.1

        if distance < min_dist:
            min_dist = distance
            #print(f'Distance: {min_dist}')

        if old_reward == 10:
           max_dist = 9_999_999
           min_dist = None

        if distance < max_dist:
            max_dist = distance
            #print(f'Distance: {max_dist}')
            #reward += 0.1
        
        state_new = agent.get_state(game)

        # train short memory
        if retrain:
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            max_dist = 9_999_999
            min_dist = None
            agent.n_games += 1
            if retrain:
                agent.train_long_memory()

            if retrain:
                if score > record:
                    record = score
                    agent.model.save(f'{model_name}.pth')

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            # Save training results
            rows = [plot_scores, plot_mean_scores]

            csv_folder_path = './results'
            filename = os.path.join(csv_folder_path, f'{model_name}.csv')
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)
            
            plot(plot_scores, plot_mean_scores, filename=os.path.join(csv_folder_path, f'{model_name}.png'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Snake Game Parameters')
    # AGENT
    # discount, max_memory, learning_rate, batch_size
    parser.add_argument('--discount', type=float, default=0.9, help='Agent discount Rate')
    parser.add_argument('--max_memory', type=int, default=100_000, help='Agent Max memory')
    parser.add_argument('--batch_size', type=int, default=1000, help='NN Batch size') 
    parser.add_argument('--learning_rate', type=float, default=0.001, help='NN Learning rate') 
    
    # GAME
    parser.add_argument('--block_size', type=int, default=20, help='Size of the snake blocks')
    parser.add_argument('--speed', type=int, default=20, help='Speed of the game')
    parser.add_argument('--width', type=int, default=640, help='Width of the game window')
    parser.add_argument('--height', type=int, default=480, help='Height of the game window')

    parser.add_argument('--max_n_games', type=int, default=1_000, help='Max number of games')

    args = parser.parse_args()

    train(args.width, args.height, args.block_size, 
          args.speed, args.discount, args.max_memory, 
          args.learning_rate, args.batch_size, args.max_n_games)