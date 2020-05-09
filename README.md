# diffsort
An implementation of the main concepts introduced in Fast Differentiable Sorting and Ranking by Blondel et al., 2020

The differentiable ranking operator is defined in full within softRank.py
A learning experiment which utilizes the operator is implemented in dnnRanks.py
Model states and training data for each experiment are saved in the .ckpt and .p files
The source file plots.py can be used to replicate training and testing curves
An implementation of differentiable sorting, still under construction, is available in fwdSort.py
