# LatticeResponse_NN_Prediction
A recurrent neural network model to predict the stress-strain curve and energy absorption of lattices given its cross section.

To run this example, first generate the input/output datasets with:
python 1_WriteInputs.py

Then, to train the auto encoder and the GRU model, run:
python 2_GRU_Model.py

To view the predictions, run:
python 3_ReadResults.py