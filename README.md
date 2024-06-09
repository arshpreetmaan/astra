# Astra: A Graph Neural Network (GNN) Decoder for QLDPC codes



A graph neural network which works on the Tanner graph of the error correcting codes. **Astra** learns to operate the belief-propagation algorithm on that graph.

Preliminary results of the decoder, when compared against Minimum Weight Perfect Matching

Some remarks

    Required python version == 3.11  
    requirement.txt is for mac m2  
    models were trained on float16 precision using nvidia gpus.

## Files
- `gnn_train.py` to train the gnn model
- `gnn_test.py` decoder using the gnn model
- `panq_functions` contains the GNN model and all the required functions
