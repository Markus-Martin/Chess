# Chess
A python version of Chess that can run with players controlling the moves or AI's.

## How does it work?
Whenever the program is run in AIvAI mode or PvAI mode, the agent is learning and this is automatically saved before the program ends. The information is also automatically loaded whenever the program is run. For best results, the AI should be trained from the game database then test it with PvAI mode. To reset whatever the AI has learned, the files that hold information should be deleted (they will be titled whatever the SAVE_NAME option is set to).

## Options
The options are located in the constants file under the heading 'options'. 

### PLAY_MODE
0 - PvAI  
1 - PvP  
2 - AIvAI
### PLAYER_STARTS
Whether the player is white or not.
### RENDER_FOR_AI
Whether to render the game for AIvAI mode
### AI_MOVE_DELAY
Movement delay for the AI actions in seconds
### USE_NEURAL_NETWORK
Whether to use the neural network (Deep Reinforcement Learning instead of Reinforcement Learning)
### AI_MODE
0 - Random Actions  
1 - QLearn by using the AI's own chosen actions (based on some exploration/exploitation strategy like epsilon greedy)  
2 - Play the best action using the information obtained so far (i.e. do not explore)  
3 - QLearn from provided games
### ALLOCATED_RUN_TIME
How long the program is allowed to run for (0 seconds means do only 1 episode). Note that the program will save every 100 episodes regardless of this parameter.
### SHOW_STATS
Whether to track and store statistics like game length in a text file for later reading.
### SAVE_NAME
Name of the file that the information for AI is saved into
### STATE_SIZE
Size of the state space (larger is more accurate but slower to learn). When using the neural network, this option decides how many past episode memories should be stored.
### LEARN_FROM
The name of the player to learn from (e.g. "Carlsen, M" to learn exclusively from Magnus Carlsen). If this value is set to None, the program won't care who it learns from. Note that due to the size of the database of chess games (4 million games), specifying a particular name may lead to long load times of the next game while the program searches for the next game with the appropriate name.
### ELO_RANGE
Elo range for which the agent will learn from. Because this is more general, the issues with LEARN_FROM aren't really present here unless an uncommon elo range is selected like 1000-1500. Note that ranges should be inputted as a tuple (1000, 1500).
### STATE_TYPE
This option is only used when in the neural network isn't being used. It decides on how the agent observes the environment because an exact representation of the state is too large. The options are:  
0 - Brute force method that doesn't attempt to capture any features in particular - just converts the chess board FEN string to a number  
1 - Represented by controlled blocks of squares. In particular, a control value of (0 neither controls, 1 agent controls, -1 enemy controls) are set for 2x4 blocks around the chess board.  
2 - Incorporates controlled squares, time frame (early, mid, late) and king safety  
