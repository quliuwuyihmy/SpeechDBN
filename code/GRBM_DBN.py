"""
"""
import cPickle
import gzip
import os
import sys
import time
import datetime
import re
from matplotlib import pyplot as plt

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from rbm import RBM
from grbm import GBRBM
from utils import zero_mean_unit_variance
from utils import normalize

import warnings
warnings.filterwarnings("ignore")

from numpy import tanh, fabs, mean, ones
from PIL import Image
from matplotlib.pyplot import hist, title, subplot

def sigmoid(xx):
    return .5 * (1 + tanh(xx / 2.))
    
from utils import tile_raster_images

class GRBM_DBN(object):

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10,
                 weight_decay=0.0002):

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.weight_decay = weight_decay

        assert self.n_layers > 0
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector
                                 # of [int] labels

        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i - 1]
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)
            # Construct an RBM that shared weights with this layer
            if i == 0:
                layer_type = GBRBM
            else:
                layer_type = RBM

            rbm_layer = layer_type(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b,
                            weight_decay=weight_decay)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # add regularization
        L2=[]
        for i in range(self.n_layers):
            L2.append((self.sigmoid_layers[i].W ** 2).sum())
        L2.append((self.logLayer.W ** 2).sum())
        self.L2 = T.sum(L2) 

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y) + 0.5 * self.weight_decay * self.L2

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

        self.oldparams = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape, dtype=theano.config.floatX)) for p in self.params]

    def pretraining_functions(self, train_set_x, batch_size, k):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for i, rbm in enumerate(self.rbm_layers):

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)
            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, momentum):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.scalar('learning_rate', dtype=theano.config.floatX)

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []

        for param, gparam, oldparam in zip(self.params, gparams, self.oldparams):
            delta = learning_rate * gparam + momentum * oldparam
            updates.append((param, param - delta))
            updates.append((oldparam, delta))

        train_fn = theano.function(inputs=[index, learning_rate],
              outputs=self.finetune_cost,
              updates=updates,
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: train_set_y[index * batch_size:
                                          (index + 1) * batch_size]})

        test_score_i = theano.function([index], self.errors,
                 givens={self.x: test_set_x[index * batch_size:
                                            (index + 1) * batch_size],
                         self.y: test_set_y[index * batch_size:
                                            (index + 1) * batch_size]})

        valid_score_i = theano.function([index], self.errors,
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        f = file(filename, 'wb')
        cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def load(filename):
        f = file(filename, 'rb')
        loaded_obj = cPickle.load(f)
        f.close()
        return loaded_obj

    def update_finetune_cost(self, weight_decay):
        self.weight_decay = weight_decay

        L2=[]
        for i in range(self.n_layers):
            L2.append((self.sigmoid_layers[i].W ** 2).sum())
        L2.append((self.logLayer.W ** 2).sum())
        self.L2 = T.sum(L2)

        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y) + 0.5 * self.weight_decay * self.L2

        self.oldparams = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape, dtype=theano.config.floatX)) for p in self.params]

def plot_weights_hists(dbn, epoch, layer_idx):
    layer = dbn.rbm_layers[layer_idx]

    weights = [x for line in layer.W.get_value() for x in line]

    plt.hist(weights)
    plt.savefig('weights\\layer_{}_weights_{}'.format(layer_idx, epoch))
    plt.clf()

    plt.hist(layer.vbias.get_value())
    plt.savefig('vbiases\\layer_{}_vbias_{}'.format(layer_idx, epoch))
    plt.clf()
        
    plt.hist(layer.hbias.get_value())
    plt.savefig('hbiases\\layer_{}_hbias_{}'.format(layer_idx, epoch))
    plt.clf()


def test_GRBM_DBN(finetune_lr=0.1, pretraining_epochs=[225, 75],
             pretrain_lr=[0.002, 0.02], k=1, weight_decay=0.0002,
             momentum=0.9, datasets=None, batch_size=128,
             hidden_layers_sizes=[1024, 1024, 1024],
             n_ins=784, n_outs=10, filename="../data/1234.pickle",
             load=True, save=True, verbose=False, pretraining_start=0,
             pretraining_stop=-1, finetune=True):

    if datasets is None:
        from load_data_MNIST import load_data
        datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)

    loaded = False

    if load:
        print '... trying to load the model'

        if os.path.isfile(filename):
            dbn = GRBM_DBN.load(filename)
            dbn.update_finetune_cost(weight_decay=weight_decay)
            loaded = True
            print '... model loaded'
        else:
            print '... couldn\' find the model file'

    if not loaded:
        print '... building the model'
        # construct the Deep Belief Network
        dbn = GRBM_DBN(numpy_rng=numpy_rng, n_ins=n_ins,
                    hidden_layers_sizes=hidden_layers_sizes,
                    n_outs=n_outs)

        #########################
        # PRETRAINING THE MODEL #
        #########################

        print '... getting the pretraining functions'
        pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size,
                                                    k=k)

        print '... pre-training the model'
        start_time = time.clock()
        ## Pre-train layer-wise

        if pretraining_stop == -1:
            pretraining_stop = dbn.n_layers

        for i in xrange(pretraining_start, pretraining_stop):
            start_time_temp = time.clock()
            if i==0:
                pretrain_lr_new = pretrain_lr[0]
                pretraining_epochs_new = pretraining_epochs[0]
            else:
                pretrain_lr_new = pretrain_lr[1]
                pretraining_epochs_new = pretraining_epochs[1]

            # go through pretraining epochs
            costs = []
            for epoch in xrange(pretraining_epochs_new):
                if verbose:
                    # weights
                    image = Image.fromarray(
                        tile_raster_images(
                            X=dbn.rbm_layers[i].W.get_value(borrow=True).T,
                            img_shape=(28, 28),
                            tile_shape=(10, 10),
                            tile_spacing=(1, 1)
                        )
                    )
                    image.save('filters\\filters_at_layer_%i_epoch_%i.png' % (i, epoch))
                    
                    # probabilities
                    X = valid_set_x[:20].eval()
                    hMean = sigmoid(numpy.dot(X, dbn.rbm_layers[i].W.get_value(borrow=True)) + dbn.rbm_layers[i].hbias.get_value(borrow=True))
                    image = Image.fromarray(hMean * 256)
                    image.save('probabilities\\probabilities_at_layer_%i_epoch_%i.gif' % (i, epoch))
                       
                    #histograms
                    plot_weights_hists(dbn, epoch, i)
                    
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                                                lr=pretrain_lr_new))
                end_time_temp = time.clock()
                mean_cost = numpy.mean(c)
                costs.append(mean_cost)
                
                print 'Pre-training layer %i, epoch %d, cost %f ' % (i + 1, epoch + 1, mean_cost) + ' ran for %d sec' % ((end_time_temp - start_time_temp) )
                
            plt.plot(costs)
            plt.savefig('layer_{}_costs'.format(i))
            plt.clf()
            
        

        end_time = time.clock()
        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

        if save:
            print '... saving the model'
            dbn.save(filename)

    ########################
    # FINETUNING THE MODEL #
    ########################

    if finetune:
        # get the training, validation and testing function for the model
        print '... getting the finetuning functions'
        train_fn, validate_model, test_model = dbn.build_finetune_functions(
                    datasets=datasets, batch_size=batch_size, momentum=momentum)

        print '... finetunning the model'

        best_params = None
        best_validation_loss = numpy.inf
        last_validation_loss = numpy.inf
        test_score = 0.
        start_time = time.clock()
        current_lr = finetune_lr
        done_looping = False
        epoch = 0

        while not done_looping:
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_fn(minibatch_index, current_lr)
                iter = (epoch - 1) * n_train_batches + minibatch_index

            import warnings
            warnings.filterwarnings("ignore")
            validation_losses = validate_model()
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, validation error %f %%' % \
                  (epoch, this_validation_loss * 100.))

            if this_validation_loss < best_validation_loss:
                best_validation_loss = this_validation_loss

            if this_validation_loss > last_validation_loss:
                current_lr /= 2.
                print(('    learning rate halved to %f') %
                      (current_lr))

            last_validation_loss = this_validation_loss

            if current_lr < 0.0001:
                done_looping = True

        test_losses = test_model()
        test_score = numpy.mean(test_losses)

        end_time = time.clock()
        print(('Optimization complete with best validation score of %f %%,'
               'with test performance %f %%') %
                     (best_validation_loss * 100., test_score * 100.))
        print >> sys.stderr, ('The fine tuning code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time)
                                                  / 60.))
        
        if save:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
            print '... saving the final model'
            dbn.save(re.sub('.pickle$', '', filename) + '_' + st + '.final.pickle')

        return (best_validation_loss * 100., test_score * 100.)

    return (0., 0.)
