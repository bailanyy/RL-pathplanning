# QLearn
This is a simple demo of Q-Learn.

`python QLearn.py` or `python display.py` to run. You will get a Q table txt file in json when you first time to run it. If you change number of rows or columns in py script, you will get new Q file.`QLearn.py` just export a command print. `display.py` has a window and more powerful display than `QLearn.py`

## 1. intruduction
This is a search E project about QLearn. I create a world(`MxN`) consisted of tiles. There is four elements including: 
1. " P " player
2. " X " trap
3. " E " end
4. "  " blank, available for player to move 

Player will move untile game over or game win. If player step on trap, game over. If player step on end, game win. The behave of player is controled by Q table. Player will excute a action that has max Q value. If player step on trap, he will get -100 scores. If player step on end, he will get 100 scores. And step on blank is -1 scores.
