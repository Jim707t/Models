from tictactoe import *
import math



X = "X"
O = "O"
EMPTY = None

board = [[X, O, EMPTY],
         [X, EMPTY, EMPTY],
         [O, EMPTY, EMPTY]]

print(minimax(board)) 
