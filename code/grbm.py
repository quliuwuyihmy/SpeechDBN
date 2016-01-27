from rbm import RBM

import theano
import theano.tensor as T

class GBRBM(RBM):
    def __init__(self, input, n_visible=784, n_hidden=500, \
                 W=None, hbias=None, vbias=None, numpy_rng=None, transpose=False,
                 theano_rng=None, weight_decay=0.0002):
            RBM.__init__(self, input=input, n_visible=n_visible, n_hidden=n_hidden, \
                         W=W, hbias=hbias, vbias=vbias, numpy_rng=numpy_rng,
                         theano_rng=theano_rng, weight_decay=weight_decay)

    def free_energy(self, v_sample):            
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = 0.5 * T.dot((v_sample - self.vbias), (v_sample - self.vbias).T)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - T.diagonal(vbias_term)
# version from Florian Metze - nie dalo dobrych wynikow
        #sq_term = 0.5 * T.sqr(v_sample - self.vbias).sum(axis = 1)
        #softplus_term = T.nnet.softplus((T.dot(v_sample, self.W) + self.hbias)).sum(axis = 1)
        #return sq_term - softplus_term

    def sample_v_given_h(self, h0_sample):
    
        '''
            Since the input data is normalized to unit variance and zero mean, we do not have to sample
            from a normal distribution and pass the pre_sigmoid instead. If this is not the case, we have to sample the
            distribution.
        '''
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # in fact, you don't need to sample from normal distribution here and just use pre_sigmoid activation instead        
        # v1_sample = self.theano_rng.normal(size=v1_mean.shape, avg=v1_mean, std=1.0, dtype=theano.config.floatX) + pre_sigmoid_v1
        v1_sample = pre_sigmoid_v1
        return [pre_sigmoid_v1, v1_mean, v1_sample]

