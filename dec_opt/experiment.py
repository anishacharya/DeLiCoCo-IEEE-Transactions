import time
from dec_opt.data_reader import DataReader
from dec_opt.dec_gd import DecGD


def run_exp(args, model):
    data_set = args.d
    root = args.r
    """ 
    Get Data specifically,  AX = Y , 
    A [samples x Feature] 
    X parameters to estimate 
    Y Target
    """
    print("loading data-set")
    t0 = time.time()
    data_reader = DataReader(root=root, data_set=data_set, download=True)
    print('Time taken to load Data {} sec'.format(time.time() - t0))

    dec_gd = DecGD(data_reader=data_reader,
                   hyper_param=args,
                   model=model)
    return dec_gd.train_losses, dec_gd.test_losses
