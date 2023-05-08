"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    num_x = sum(row.count(X) for row in board)
    num_o = sum(row.count(O) for row in board)
    
    if num_x <= num_o:
        return X
    else:
        return O 
         
         
def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    moves = set()
    
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == EMPTY:
                moves.add((i, j))
                
    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if not action in actions(board):
        raise Exception("No valid action")
    new_board = copy.deepcopy(board)
    i, j = action
    new_board[i][j] = player(board)
    
    return new_board
    

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows
    for row in board:
        if row == [X, X, X]:
            return X
        elif row == [O, O, O]:
            return O

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col]:
            if board[0][col] == X:
                return X
            elif board[0][col] == O:
                return O

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2]:
        if board[0][0] == X:
            return X
        elif board[0][0] == O:
            return O
    elif board[0][2] == board[1][1] == board[2][0]:
        if board[0][2] == X:
            return X
        elif board[0][2] == O:
            return O

    return None


def is_tie(board):
    if all(cell != EMPTY for row in board for cell in row):
        return True
    return False


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    
    if is_tie(board):
        return True
    
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    elif is_tie(board):
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    
    if player(board) == X:
        return max_value(board)[1]
    elif player(board) == O:
        return min_value(board)[1]
            
    
def max_value(board):
    if terminal(board):
        return utility(board), None
    
    v = -math.inf
    best_action = None
    
    for action in actions(board):
        value = min_value(result(board, action))[0]
        if value > v:
            v = value
            best_action = action
    
    return v, best_action
        
        
def min_value(board):
    if terminal(board):
        return utility(board), None
      
    v = math.inf
    best_action = None
    
    for action in actions(board):
        value = max_value(result(board, action))[0]
        if value < v:
            v = value
            best_action = action
    
    return v, best_action     
         