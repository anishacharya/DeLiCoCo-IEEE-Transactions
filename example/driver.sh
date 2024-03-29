python3 driver.py --d 'mnist' --o 'mnist/paper/Q' --algorithm 'ours' --n_cores 9 \
--topology 'torus' \
--Q 10 --consensus_lr 0.9 --quantization_function 'top' --fraction_coordinates 0.5 --epochs 5000 -\
-initial_lr 0.2 --n_repeat 1 --num_bits 2 --dropout_p 0.5 --regularizer 0 \
--task 'log_reg'

