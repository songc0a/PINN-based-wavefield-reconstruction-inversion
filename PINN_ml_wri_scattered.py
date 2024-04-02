"""
@author: Chao Song
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
import cmath
from functions_augmented_atan import PhysicsInformedNN
from functions_velocity_inversion import Velocityinversion

np.random.seed(1234)
tf.set_random_seed(1234)

fre = 4.0
PI = 3.1415926
omega = 2.0*PI*fre
niter_du = 50000
niter_v = 10000
nz = 101
nx = 401
dx = 0.016
dz = 0.016
ns = 10
N_train = nx*nz*ns

outerniter = 4

if __name__ == "__main__":

########### solving augmented wave equation

    layers = [3, 64, 64, 32,32,16,16,8,8, 4]
    layers1 = [2, 20, 20, 20, 20, 20, 1]

   # Load Data
    data = scipy.io.loadmat('sigsbee_4hz_star_multisource.mat')
    du = data['dU'] # true scattered wavefield of real part
  
    x = data['x_coor'] # x coordinate
    z = data['y_coor'] # y coordinate

    x_test = data['x_test'] # x coordinate
    z_test = data['y_test'] # y coordinate
    sx_test = data['sx_test'] # y coordinate

    u0 = data['U0'] 
    m = data['m'] # N=10000
    m0 = data['m0'] # N=10000

    N = u0.shape[0]
    m_star = m
   # Training Data

  #  train_data = scipy.io.loadmat('marmousi_4hz_test_random_multisource_full_N20000.mat')
    train_data = scipy.io.loadmat('sigsbee_4hz_test_random_multisource_G2D_N10000.mat')

    u0_train_aug = train_data['U0_train'] # background wavefield
    du_train_aug = train_data['dU_train'] # scattered wavefield
  
    x_train_aug = train_data['x_train'] # x coordinate for training
    z_train_aug = train_data['y_train'] # y coordinate for training
    sx_train_aug = train_data['sx_train'] # y coordinate for training

    m_train_aug = train_data['m_train'] # true squared slowness
    m0_train_aug = train_data['m0_train'] # background squared slowness

    model = PhysicsInformedNN(x_train_aug, z_train_aug, sx_train_aug, u0_train_aug, du_train_aug, m_train_aug, m0_train_aug, layers, omega)   

    for it in range(outerniter):
        # Training for solveing augmented wave equation
        if it == 0:
           niter_du = 20000
        else:
           niter_du = 20000

   #     model = PhysicsInformedNN(x_train_aug, z_train_aug, sx_train_aug, u0_train_aug, du_train_aug, m_train_aug, m0_train_aug, layers, omega)
        model.train(niter_du)

        # Prediction
        du_pred, du_xx_pred, du_zz_pred = model.predict(x_test, z_test, sx_test)

        scipy.io.savemat('du_pred_4hz_atan_l64_niter%d_N10_b6.mat'%(it+0),{'du_pred':du_pred})
      #  scipy.io.savemat('du_pred_4hz_sine_l9_b705_mean_N20000_niter%d.mat'%(it),{'du_pred':du_pred})

        # Training Data
        idx = np.random.choice(N, N_train, replace=False)
        x_train = x_test[idx,:]
        z_train = z_test[idx,:]

        u0_train = u0[idx,:]
        du_train = du_pred[idx,:]
        
        du_xx_train = du_xx_pred[idx,:]
        du_zz_train = du_zz_pred[idx,:]
        m0_train = m0[idx,:]

        # Training
        model1 = Velocityinversion(x_train, z_train, u0_train, du_train, du_xx_train, du_zz_train, m0_train, layers1, omega)
        model1.train(niter_v)

        ################### Test Data

        # Prediction
        m_pred = model1.predict(x, z)
        m_train_aug = model1.predict(x_train_aug, z_train_aug)
        
        model.update_v(m_train_aug)

       # scipy.io.savemat('m_pred4hz_sine_l9_b705_N20000_mean_niter%d_sequential.mat'%(it),{'m_pred':m_pred})
        scipy.io.savemat('m_pred4hz_atan_l64_niter%d_N10_b6.mat'%(it+0),{'m_pred':m_pred})
    ######################################################################
    ######################## Save files  #################################
    ######################################################################
 #   scipy.io.savemat('du_pred_3hz_G2D_n128_niter1_atan_b701.mat',{'du_pred':du_pred}) 
 #   scipy.io.savemat('du_star_full_3hz.mat',{'du_star':du})

#    scipy.io.savemat('m_pred_3hz_G2D_n128_linear_atan_b701.mat',{'m_pred':m_pred})
  #  scipy.io.savemat('m_mar_full.mat',{'m_true':m_star})




