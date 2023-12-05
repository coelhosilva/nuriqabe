from nuriqabe.board_instances import boards_summary

# Pandas DataFrame with id, shape, size, whether it is a square board, and n_islands
df_boards = boards_summary()

print(df_boards)
