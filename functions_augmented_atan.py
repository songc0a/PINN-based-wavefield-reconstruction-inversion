import tensorflow as tf
import numpy as np
import time
import cmath
import scipy.io

nz = 101
nx = 401
ns = 10
N_train = 10000*ns
misfit = []
misfit1 = []

def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x

####### Class for augmented wave equation
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, z, sx, u0, du, m, m0, layers, omega):
        
        X = np.concatenate([x, z, sx], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        
        self.x = X[:,0:1]
        self.z = X[:,1:2]
        self.sx = X[:,2:3]


        self.u0 = u0
        self.m = m
        self.m0 = m0
        self.du = du
        self.layers = layers
        self.omega = omega
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)  

        # tf placeholders 
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.sx_tf = tf.placeholder(tf.float32, shape=[None, self.sx.shape[1]])

        self.du_tf = tf.placeholder(tf.complex64, shape=[None, self.du.shape[1]])
       
        self.du_pred, self.f_loss, self.du_xx_pred, self.du_zz_pred = self.net_NS(self.x_tf, self.z_tf, self.sx_tf)

        #self.alpha = tf.Variable(tf.zeros([1,1], dtype=tf.float32)+0.0, dtype=tf.float32)

        # loss function we define 0.0000005 
        self.loss = 0.0000006*tf.reduce_sum(tf.square(tf.abs(self.f_loss))) + \
                    tf.reduce_sum(tf.square(tf.abs(self.du_tf[N_train:] - self.du_pred[N_train:]))) 
                    
                    
        # optimizer used by default (in original paper)        
            
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 100000,
                                                                           'maxfun': 100000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32)+0.0, dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.atan(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS(self, x, z, sx):
    
        # output scattered wavefield: du_real du_imag, loss function: L du+omega^2 dm u0 

        omega = self.omega
        m = self.m
        m0 = self.m0
        u0 = self.u0

        dureal_and_duimag = self.neural_net(tf.concat([x,z,sx], 1), self.weights, self.biases)

        du_real = dureal_and_duimag[:,0:1]
        du_imag = dureal_and_duimag[:,1:2]
        du = tf.complex(du_real, du_imag)

        dudx = fwd_gradients(du, x)
        dudz = fwd_gradients(du, z)

        du_xx = fwd_gradients(dudx, x)
        du_zz = fwd_gradients(dudz, z)
        
        f_loss =  omega*omega*m*du + du_xx + du_zz + omega*omega*(m-m0)*u0 #  L du + omega^2 dm u0 
        return du, f_loss, du_xx, du_zz

    
    def callback(self, loss):
        print('Loss: %.3e' % (loss))
        misfit1.append(loss)
        scipy.io.savemat('misfit1_4hz_atan_niter1_l64_N10_b6_ori.mat',{'misfit1':misfit1})      


    def train(self, nIter): 

        tf_dict = {self.x_tf: self.x, self.z_tf: self.z, self.sx_tf: self.sx, self.du_tf: self.du}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            misfit.append(loss_value)         
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e,Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
        scipy.io.savemat('misfit_4hz_atan_niter1_l64_N10_b6_ori.mat',{'misfit':misfit})
            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

            
    
    def predict(self, x_star, z_star, sx_star):
        
        tf_dict = {self.x_tf: x_star, self.z_tf: z_star, self.sx_tf: sx_star}
        
        du_star = self.sess.run(self.du_pred, tf_dict)
        du_xx_star = self.sess.run(self.du_xx_pred, tf_dict)
        du_zz_star = self.sess.run(self.du_zz_pred, tf_dict)

        return du_star, du_xx_star, du_zz_star


    def update_v(self, m_new):

        self.m = m_new
