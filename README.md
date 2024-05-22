# Graph Neural Network (GNN) Decoder

A graph neural network which works on the tanner graph of error correcting code. It learns to operate BP algorithm on that graph.  
Some remarks

    Required python version == 3.11  
    requirement.txt is for mac m2  
    models were trained on float16 precision using nvidia gpus.

## Files
- `gnn_train.py` to train the gnn model
- `gnn_test.py` decoder using the gnn model
- `panq_functions` contains the GNN model and all the required functions
