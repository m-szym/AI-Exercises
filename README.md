# AI Exercises
### Artificial Intelligence laboratory
## Author: Marek Szymański

## Project description
This repository holds various small exercises done while exploring the Artificial Intelligence course. Wide range of AI topics are covered, from linear regression and minimax to neural networks and decision trees.

## Project map
1. simple-linear-regression - implements basic linear regression algorithm from scratch and uses it to predict MPG for various cars
2. genetic-algorith - implements a genetic algorithm with 1-point crossover and roulette-wheel selection. It is used to solve the classic knapsack problem (at 2 knapsack sizes) and compared with brute-force solution
3. minimax-alphabeta - implements a number of automated agents capable of playing a simple connect-4-symbols-before-enemy game. Simple minimax and minimax with alpha-beta pruning agents are implemented and evaluated against each other and the baseline of fully random agent. Option for human-ai game is also available
4. decision-trees-forests - implements random decision tree for decision making and also forest of decision trees for group decision making. Dataset of Titanic passangers and their fates is then used to train and test the effectiveness of trees / forests
5. kmeans - implements a clustering alogrithm using kmeans, with forgy or kmeans++ centroid initialization. The classic Iris dataset is used to test the clustering algorithm
6. neural-nets - implements 2 simple neural networks: one from scratch with numpy and the other in pytorch. The former is tested on AND and XOR logic gates, while the latter performs classification of 2 interwoven spirals on 2D plane
7. reinforcement-learning - implements q-agent with epsilon-greedy algorithm and trains it on the [frozen lake environment[(https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) 
