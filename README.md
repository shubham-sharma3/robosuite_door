# Robosuite Door RL Implementation
Repository for training robot manipulator to open a door in robosuite env. 

Several algorithm were implemented, including TD3, PPO, SAC. However, I only managed to make TD3 agent successfully open the door.

Each folder contain python scripts to train the agent to drive the robot. tmp folder contains the path file for training process and logs folder for tensorboard messages.

## Dependencies
Run following command to install necessary libs and dependencies to run the code. **It's highly recommended to install them in conda environment or virtual environments.** 
```angular2html
pip3 install -r requirements.txt
```
Pytorch is needed to train the agent and you should reference to pytorch.org to install correct version of torch alone with your CUDA. My training process was done via torch 1.11 + cu102 on NVIDIA RTX TITAN GPU.

## How to use
Run main scripts in each folder for each algorithm implementation. As only self implemented TD3 is working, you can run the script in folder TD3/main.py to train the model. 
After around 1k steps it will converge an successfully teach the robot to open the door. Run test.py script in the same folder to visualize the robot implemented trained model. 
