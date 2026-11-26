def show(board):
    for i in range(0, 9, 3):
        print(" | ".join(board[i:i+3]))
        if i < 6: print("--+---+--")

def winner(board, mark):
    lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    return any(all(board[i] == mark for i in line) for line in lines)

board = [' '] * 9
player = 'X'
for turn in range(9):
    show(board)
    try:
        pos = int(input(f"Player {player}, choose 1-9: ")) - 1
    except ValueError:
        print("Enter a number."); continue
    if pos not in range(9) or board[pos] != ' ':
        print("Invalid move."); continue
    board[pos] = player
    if winner(board, player):
        show(board); print(player, "wins!"); break
    player = 'O' if player == 'X' else 'X'
else:
    show(board); print("Draw!")
