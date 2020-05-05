python3 driver.py --d 'mnist_partial' --algorithm 'ours' --n_cores 9 --topology 'ring' --Q 1 --consensus_lr 1.0 \
--quantization_function 'top' --fraction_coordinates 1.0 --epochs 5000 --initial_lr 0.2 --n_repeat 1

# python3 driver.py --d 'mnist_partial' --algorithm 'baseline' --n_cores 1 --initial_lr 0.2 --epochs 5000 --n_repeat 1