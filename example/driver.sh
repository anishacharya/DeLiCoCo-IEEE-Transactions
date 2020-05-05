python3 driver.py --d 'mnist_partial' --algorithm 'ours' --n_cores 9 --topology 'ring' --Q 5 --consensus_lr 0.5 \
--quantization_function 'top' --fraction_coordinates 0.5 --epochs 5000 --initial_lr 0.2 --n_repeat 1

# python3 driver.py --d 'mnist_partial' --algorithm 'baseline' --n_cores 1 --initial_lr 0.2 --epochs 5000 --n_repeat 1