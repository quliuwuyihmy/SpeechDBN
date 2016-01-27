
from GRBM_DBN import test_GRBM_DBN
from load_data_MNIST import load_data


datasets = load_data()

test_GRBM_DBN(finetune_lr=0.2, pretraining_epochs=[200,50],
    pretrain_lr=[0.0001, 0.002], k=1, weight_decay=0.002,
    momentum=0.7, batch_size=20, datasets=datasets,
    hidden_layers_sizes=[784,784], load=False, save=True,
    filename='../data/MNIST/GRBM200/layer-4.pickle',
    finetune=True, pretraining_start=0, pretraining_stop=2, verbose=True)
