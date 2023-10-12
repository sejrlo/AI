from sentdexNeuralnetwork import *
import os
import Kryds_Bolle_I_Kryds_Bolle as game
import numpy as np
import time

model_path = "tic_tac_toe_AI"


if os.path.isfile(model_path):
    model = Neural_Network.load(model_path)

else:
    model = Neural_Network(
        [
            Layer_Dense(81, 200),
            Activation_ReLU(),
            Layer_Dense(200, 200),
            Activation_ReLU(),
            Layer_Dense(200,81),
            Activation_Softmax(),
            
        ],
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(),
        accuracy=Accuracy_Categorical(binary=False),
        )

model.save(model_path)


while game.running:
    space = model.predict(np.array([game.board.get_board_stat()]))[0]
    success = game.board.place_piece_on_board(space)
    if not success:
        for space in reversed(np.argsort(model.output[0])):
            success = game.board.place_piece_on_board(space)
            if success: break

print(game.board.resolved)



