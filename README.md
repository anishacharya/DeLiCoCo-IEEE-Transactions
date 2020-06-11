# FAQ : De-LiCoCo
 
**How do I run the Code?**
```
A. Install our package: 
pip3 install decopt

(A.1) Often get the latest update:
 pip3 install decopt --upgrade 

B. Get Data: 
sh pull_data.sh breast_cancer

c. Run script with default parameters: 
python3 driver.py

With different parameters:
python3 driver.py --d 'mnist' --n_cores 10 --algorithms 'ours'


Parameter Options:

    parser.add_argument('--d', type=str, default='breast_cancer',
                        help='Pass data-set')
    parser.add_argument('--task', type=str, default='log_reg', help='Choose task')
    parser.add_argument('--r', type=str, default=os.path.join(curr_dir, './data/'),
                        help='Pass data root')
    parser.add_argument('--stochastic', type=bool, default=False)
    parser.add_argument('--algorithm', type=str, default='ours')

    parser.add_argument('--n_cores', type=int, default=9)

    parser.add_argument('--topology', type=str, default='ring')
    parser.add_argument('--Q', type=int, default=2)
    parser.add_argument('--consensus_lr', type=float, default=0.3)

    parser.add_argument('--quantization_function', type=str, default='full')
    parser.add_argument('--num_bits', type=int, default=2)
    parser.add_argument('--fraction_coordinates', type=float, default=0.1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_type', type=str, default='constant')
    parser.add_argument('--initial_lr', type=float, default=0.01)
    parser.add_argument('--epoch_decay_lr', type=float, default=0.9)
    parser.add_argument('--regularizer', type=float, default=0)

    parser.add_argument('--estimate', type=str, default='final')
    parser.add_argument('--n_proc', type=int, default=3, help='no of parallel processors for Multi-proc')
    parser.add_argument('--n_repeat', type=int, default=3, help='no of times repeat exp with diff seed')
```

**Supported argument values**
```
    --d : 'breast_cancer' || 'mnist' || 'syn1' || 'syn2'
    --task : 'log_reg' || 'lin_reg' || 'nlin_reg'
```

**How do I reproduce the plots in the paper?**
```
Check plots.py
Ex. MNIST
It has clearly marked code to run Fig1, Fig2, FIg3 for mnist
```

**Where are the results stored ?**
```
Ex. For MNIST
The results of mnist experiments are stored in results/mnist_partial.
There are 3 folders, Q means experiments with Q, C is Compression, T is Topology.
The results and parameters are stored as pickle file
and can be readily consumed by plots.py
```

**How do I reproduce the results ?**
```
Ex. MNIST
For all experiments since the parameters are stored in the results folder. (See above FAQ) 
please run driver.py with these parameters to produce results.
The results will automatically be stored in pickle files in appropriately 
marked folders along with corresponding parameters. 
```
