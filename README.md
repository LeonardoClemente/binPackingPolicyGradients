# binPackingPolicyGradients

See PDF for more information. (Currently working on this)

Based on recent deepmind work on RL, the idea of developing an hyper-heuristic model arises. It is possible to see an NP-hard problem as a game, where the the RL agent has access to information via hand-coded observation vectors and it plays using the heuristics as the possible actions within the game.


An hyper-heuristic selection model for binary and 4-heuristic selection was developed using the policy gradients technique. Models were trained to perform on a sub-domain of the 1 dimensional bin packing problem. Policy function is approximated using a fully connected network. Environment, score function and heuristics are hand coded and model is implemented using python and tensorflow. Results for the model with a fully connected layer show that the model does learn to take better decisions within time based on arbitrary metrics (average space used per bin), yet it is not capable to beat the individual use of heuristics.
