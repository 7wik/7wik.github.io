---
layout: post
title: Monte Carlo Tree Search
---

Hi!

This blog is briefly discusses about Monte Carlo Tree Search shortly called as MCTS, a heuristic search algorithm.

As it's name suggests, MCTS is an algorithm which is used for finding the best sequence of nodes in a tree for maximizing the outcome of an experiment/trial. This outcome is commonly called as "reward". 

The most common example where MCTS is used is the Tic-Tac-Toe game. In a 2-playered Tic-Tac-Toe game, with one intelligent player, we expect to develop a player controlled by AI which is equally competent with an intelligent player(human). In Tic-Tac-Toe's context, the AI-player must pick the best move(best child-node) among the possible moves(from the child nodes) in every chance of his to get a success as his end-result(victory). Hence, this is called a tree-search problem.

The algorithm is primarily dependent on the 4 steps mentioned below:-
1. Selection: This step involves selecting the best node from the children
2. Expansion: In this step, we add permitted(only the possible ones) child nodes to a leaf node
3. Rollout/simulation: In this step, we perform simulation by selecting nodes randomly and proceed till the end of the game
4. Backpropogation: In this step, we add the final outcome of the game to all the nodes in the path that led to that particular outcome. (This is not to be confused with the backprop in gradient based learning of neural networks)
 In every game, we use these 4 steps to train our AI-player.











