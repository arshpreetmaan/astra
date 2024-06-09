# Astra: A Graph Neural Network (GNN) Decoder for QLDPC codes



A graph neural network which works on the Tanner graph of the error correcting codes. **Astra** learns to operate the belief-propagation algorithm on that graph.

Preliminary results of the decoder, when compared against Minimum Weight Perfect Matching (MWPM), for decoding surface codes up to distance 9 affected by code capacity noise.

For the GNN we observe a threshold of 17\%: left) data obtained by numerical simulations; b) curves representing the asymptotes fitted to the numerical data.

![asta_res.png](astra_res.png)

**Files**
- `gnn_train.py` to train the gnn model
- `gnn_test.py` decoder using the gnn model
- `panq_functions` contains the GNN model and all the required functions

**Notes**
- Required Python version == 3.11  
- requirement.txt is for Mac M2  
- models were trained on Float16 precision using Nvidia GPUs

**This research was performed in part with funding from the Defense Advanced Research Projects Agency (under the Quantum Benchmarking (QB) program under award no. HR00112230006 and HR001121S0026 contracts).**
