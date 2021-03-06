# Chess
A python version of Chess that can run with players controlling the moves or AI's.

## How does it work?
Whenever the program is run in AIvAI mode or PvAI mode, the agent is learning and this is automatically saved before the program ends. The information is also automatically loaded whenever the program is run. For best results, the AI should be trained from the game database then test it with PvAI mode. To reset whatever the AI has learned, the files that hold information should be deleted (they will be titled whatever the SAVE_NAME option is set to).  
Note that the game database cannot be uploaded here due to its large size of 3GB; however, it was obtained from http://caissabase.co.uk/ (which was converted to a .pgn file).

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
Whether to track and store statistics on learning progress. The format is:  
Episode, Number of half moves, 1-step reward, 2-step reward, Percent of rewards better than just the move cost  
  
To elaborate, the 1-step and 2-step rewards take the average of the sigmoid of rewards in each episode. So, the best score approaches 1. Then the 1-step just means the immediate reward while 2-step refers to the reward after the opponent has taken their turn too. On 1-step reward, the lowest score approaches expit(-1) because the move cost is -1 and you cannot really get worse than moving without gaining anything in 1-step rewards.
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

## References
Much of the neural network was made using Dr. Phil's video as reference: https://www.youtube.com/watch?v=wc-FxNENg9U.  
The banner was made with https://www.canva.com/. Finally, it's difficult to find where the images originated (as they are copied everywhere) from but the chess pieces were obtained from free clipart websites.
### Dependencies
The Chess game uses the follow python libraries:  
pygame - used to render the game  
numpy - used in developing the neural network  
torch - used in developing the neural network  
chess - used to interpret pgn files
