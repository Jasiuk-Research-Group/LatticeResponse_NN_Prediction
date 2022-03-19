#################################################  User Inputs  #######################################################################
EPOCH = 150
train_fraction = 0.8
timesteps, input_dim = 50, 128 + 6
x_scale_list = [ 0 , 1 , 2 , 3 , 4 ]
output_list = [ 0 , 1 , 2 , 3 ]
out_key = [ 'Stress strain curve' , 'Plastic dissipation' , 'Damage disspation' , 'Elastic strain energy' ]
n_output_channels = len(output_list)

# Image autoencoder setting
latent_dim = 100
autoencoder_epoch = 80
auto_size = 10000
retrain_img_encoder = True
#################################################  User Inputs  #######################################################################



import tensorflow as tf
from tensorflow.keras.layers import Dense, TimeDistributed, GRU , concatenate, BatchNormalization , Conv2D , MaxPooling2D , Flatten , Reshape , RepeatVector, LeakyReLU
from tensorflow.keras import Input, Model
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
# from keras.utils import multi_gpu_model
from sklearn.utils import shuffle
import os
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from sklearn.preprocessing import StandardScaler , MinMaxScaler , RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
import pickle
import time
import keras.backend as K

All_train_time = []
for REP in range( 1 ):
    print('Repetition ' , REP+1 , '.....')
    # Create output dir
    try:
        os.mkdir('Repetition'+str(REP))
    except:
        pass

    #################################################  Read Inputs  #######################################################################
    with open('ImgDict.pkl', 'rb') as f:
        img_dict = pickle.load(f)
    f = open( 'InputKeys.npy' , 'rb' )
    Input_keys = np.load( f )
    f.close()
    f = open( './Inputs2.npy' , 'rb' )
    x_arry = np.load( f )
    f.close()
    f = open( './Outputs.npy' , 'rb' )
    y = np.load( f )
    f.close()

    # # Brand-new test geometries
    # with open('ImgDictNewGeom.pkl', 'rb') as f:
    #     img_dict_new = pickle.load(f)
    # f = open( './Inputs2NewGeom.npy' , 'rb' )
    # x_arry_new = np.load( f )
    # f.close()
    # f = open( './OutputsNewGeom.npy' , 'rb' )
    # y_new = np.load( f )
    # f.close()


    # Filter outputs
    y = y[:,:,output_list]
    print('Model outputs are: ')
    for i in output_list:
        print( out_key[ i ] )


    print('\nTotal data dimensions:')
    print( x_arry.shape , y.shape  )
    num_examples = x_arry.shape[0]

    # Read parameters
    f = open( './InputKeys.npy' , 'rb' )
    All_Keys = np.load( f )[:num_examples]
    f.close()

    unique_keys = np.array( list(set( All_Keys )) )
    N_unique = len(unique_keys)
    unseen_structure = unique_keys[ np.random.choice( np.arange(N_unique) , N_unique - 600 , replace=False ) ]
    print( 'Chose ' + str(len(unseen_structure)) + ' structures as testing set' )

    # Pick out those data
    flag = np.zeros( num_examples , dtype=bool )
    for i in range(num_examples):
        if All_Keys[i] in unseen_structure:
            flag[i] = True

    # Choose new data
    test2_img_key = Input_keys[ flag ]
    x_test2_arry = x_arry[ flag , : , :]
    y_test_gt2 = y[ flag , : , :]

    # Get the remaining data
    nF = np.logical_not( flag )
    img_keys = Input_keys[ nF ]
    x_arry = x_arry[ nF , : , :]
    y = y[ nF , : , :]
    img_keys, x_arry , y = shuffle(img_keys, x_arry , y)

    # Split data
    train_range    = int(num_examples * train_fraction)
    train_img_key        = img_keys[:train_range]
    x_train_arry        = x_arry[:train_range,:,:]
    y_train        = y[:train_range,:,:]
    print('Training data dimensions:')
    print(x_train_arry.shape , y_train.shape)

    test_img_key           = img_keys[train_range:]
    x_test_arry           = x_arry[train_range:,:,:]
    y_test_gt        = y[train_range:,:,:]
    print('Testing data dimensions:')
    print(y_test_gt.shape)
    print('New structure test data dimensions:')
    print(y_test_gt2.shape)



    #################################################  Image autoencoder  #######################################################################
    class Autoencoder(Model):
      def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          layers.Dense(latent_dim, activation='relu'),
          layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
          layers.Dense(128*128, activation='sigmoid'),
          layers.Reshape((128, 128))
        ])

      def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    N_eval = 200
    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection ) / (K.sum(y_true_f) + K.sum(y_pred_f) )
    if retrain_img_encoder:
        new_repeat = 0
        auto_size -= new_repeat * 3
        # Train encoder
        print('\n\nTraining image autoencoder with ' + str(auto_size) + ' images, latent dimension = ' + str(latent_dim) )

        # Build training set
        img_auto_train = np.zeros([auto_size+new_repeat * 3,128,128])
        for ii in range( auto_size ):
            curr_key = train_img_key[ii]
            img_auto_train[ii,:,:] = img_dict[ curr_key ]
        
        # # Add new design geometries to training set
        # cc = 0
        # for k , v in img_dict_new.items(): 
        #     for ii in range(new_repeat):
        #         img_auto_train[auto_size+cc,:,:] = v
        #         cc += 1


        autoencoder = Autoencoder(latent_dim)
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError() , metrics=[dice_coef] )

        m_his =autoencoder.fit(img_auto_train, img_auto_train,
                        epochs=autoencoder_epoch,
                        shuffle=True,
                        validation_split=0.15,
                        batch_size = 50 )
        # Save current encoder
        autoencoder.save('Repetition'+str(REP)+"/ImgEncoder",save_format="tf")
        print('Saved image encoder!')

        # Plot the loss and metric
        kk = list(m_his.history.keys())
        np_loss_history = np.array( m_his.history[ kk[0] ] )
        np_v_loss_history = np.array( m_his.history[ kk[2] ] )
        np_metric_history = np.array( m_his.history[ kk[1] ] )
        np_val_metric_history = np.array( m_his.history[ kk[3] ] )

        plt.figure()
        plt.plot( np_loss_history , 'tab:orange' )
        plt.plot( np_v_loss_history , 'tab:purple' )
        plt.xlabel('Training epoches')
        plt.ylabel('Loss (MSE)')
        plt.legend(['Training','Validation'])


        plt.figure()
        plt.plot( np_metric_history , 'tab:orange' )
        plt.plot( np_val_metric_history , 'tab:purple' )
        plt.xlabel('Training epoches')
        plt.ylabel('Metric (DSC)')
        plt.legend(['Training','Validation'])

        plt.show()


        # exit()

    else:
        # Just read the saved encoder
        autoencoder = tf.keras.models.load_model( 'Repetition'+str(REP)+"/ImgEncoder" , custom_objects={'dice_coef': dice_coef} )
        print('Using saved image encoder!')

    # Sanity check
    # Build validation sets
    img_auto_test = np.zeros([N_eval,128,128])
    img_auto_test2 = np.zeros([N_eval,128,128])
    for ii in range( N_eval ):
        curr_key = test_img_key[ii]
        img_auto_test[ii,:,:] = img_dict[ curr_key ]
        curr_key2 = test2_img_key[ii]
        img_auto_test2[ii,:,:] = img_dict[ curr_key2 ]

    # # Build new geom sets
    # img_auto_test_new = np.zeros([3,128,128])
    # ii = 0
    # for k , v in img_dict_new.items():
    #     img_auto_test_new[ii,:,:] = v
    #     ii += 1

    print('\nEvaluate image autoencoder:')
    model_evaluation = autoencoder.evaluate(x=img_auto_test , y=img_auto_test, batch_size= 1 )
    print('On a new strucutre:')
    model_evaluation = autoencoder.evaluate(x=img_auto_test2 , y=img_auto_test2, batch_size= 1 )
    # print('On a three new test strucutres:')
    # model_evaluation = autoencoder.evaluate(x=img_auto_test_new , y=img_auto_test_new, batch_size= 1 )

    # Plot images
    encoded_imgs = autoencoder.encoder(img_auto_test2).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    n = 5
    plt.figure()
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(img_auto_test2[i*9])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i*9])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()



    print('\n\nEncoding all image inputs now...')
    nImages = len(list(img_dict.keys()))
    All_img_raw = np.zeros([nImages,128,128])
    ii = 0
    inverse_map = {}
    for k , v in img_dict.items():
        inverse_map[ k ] = ii
        All_img_raw[ ii , : , : ] = v
        ii += 1
    All_img_encoded = autoencoder.encoder( All_img_raw )


    # # New designs
    # inverse_map2 = {}
    # All_img_raw_new = np.zeros([3,128,128])
    # ii = 0
    # for k , v in img_dict_new.items():
    #     inverse_map2[ k ] = ii
    #     All_img_raw_new[ ii , : , : ] = v
    #     ii += 1
    # All_img_encoded_new = autoencoder.encoder( All_img_raw_new )


    print('Building encoded image inputs now...')
    def fillVec( k , inverse_map , latent_dim , All_img_encoded ):
        l0 = len(k)
        out = np.zeros( [l0,latent_dim] )
        for ii in range(l0):
            curr_key = k[ii]
            idx = inverse_map[ curr_key ]
            out[ ii , : ] = All_img_encoded[ idx , : ]
        return out


    x_train_img_encoded = fillVec( train_img_key , inverse_map , latent_dim , All_img_encoded )
    x_test_img_encoded = fillVec( test_img_key , inverse_map , latent_dim , All_img_encoded )
    x_test2_img_encoded = fillVec( test2_img_key , inverse_map , latent_dim , All_img_encoded )

    # newKey = np.ones(150*50)
    # newKey[ 50*50 : 100*50 ] = 2
    # newKey[ 100*50 : ] = 3
    # x_img_encoded_new = fillVec( newKey , inverse_map2 , latent_dim , All_img_encoded_new )
    # print( x_train_img_encoded.shape , x_test_img_encoded.shape , x_test2_img_encoded.shape , x_img_encoded_new.shape )


    #################################################  Scale Inputs  #######################################################################
    # Scale datasets
    xScalers = []
    for ss in x_scale_list:
        curr_scaler = StandardScaler()
        curr_scaler.fit( x_train_arry[:,:,ss] )
        xScalers.append( curr_scaler )

        # Transform array inputs
        x_train_arry[:,:,ss] = curr_scaler.transform( x_train_arry[:,:,ss] )
        x_test_arry[:,:,ss] = curr_scaler.transform( x_test_arry[:,:,ss] )
        x_test2_arry[:,:,ss] = curr_scaler.transform( x_test2_arry[:,:,ss] )
        # x_arry_new[:,:,ss] = curr_scaler.transform( x_arry_new[:,:,ss] )


    # Transform y data
    yScalers = []
    for ss in range(n_output_channels):
        curr_scaler = StandardScaler()
        curr_scaler.fit( y_train[:,:,ss] )
        yScalers.append( curr_scaler )

        # Transform
        y_train[:,:,ss] = curr_scaler.transform( y_train[:,:,ss] )
        y_test_gt[:,:,ss] = curr_scaler.transform( y_test_gt[:,:,ss] )
        y_test_gt2[:,:,ss] = curr_scaler.transform( y_test_gt2[:,:,ss] )
        # y_new[:,:,ss] = curr_scaler.transform( y_new[:,:,ss] )



    # Transform encoded inputs
    encoded_scaler = StandardScaler()
    encoded_scaler.fit( x_train_img_encoded )

    x_train_img_encoded = encoded_scaler.transform( x_train_img_encoded )
    x_test_img_encoded = encoded_scaler.transform( x_test_img_encoded )
    x_test2_img_encoded = encoded_scaler.transform( x_test2_img_encoded )
    # x_img_encoded_new = encoded_scaler.transform( x_img_encoded_new )


    # Finally, store all scalers
    f = open('Repetition'+str(REP)+'/Scalers.npy','wb')
    np.save( f , np.array([ xScalers , encoded_scaler , yScalers ],dtype=object) )
    f.close()


    xtrain1_full = x_train_img_encoded.copy()
    xtrain2_full = x_train_arry.copy()
    y_full = y_train.copy()
    s = xtrain1_full.shape
    print('Current training size ' , s[0] )
    pct_count = 0
    # for pct in [ 0.1 , 0.2 , 0.4 , 0.6 , 0.8 ]:
    for pct in [ 1. ]:
        pct_count += 1
        print('\n\nUsing ' , pct*100 , '% of total training data in training!')
        SIZE = int(round(pct*s[0]))
        print(SIZE)
        x_train_img_encoded = xtrain1_full[ : SIZE ]
        x_train_arry = xtrain2_full[ : SIZE ]
        y_train = y_full[ : SIZE ]




        #################################################  Build and train model  #######################################################################
        # Image part
        i1 = Input( shape=( latent_dim ) )
        img = RepeatVector( timesteps )(i1)

        # Get array input
        i2 = Input(shape=( timesteps, 6 ))

        # Combine both inputs
        i = concatenate([img, i2])

        # GRU layers
        o = GRU(300, return_sequences=True , activation='tanh' )(i)
        o = GRU(300, return_sequences=True , activation='tanh' )(o)
        o = GRU(300, return_sequences=True , activation='tanh' )(o)
        o = TimeDistributed( Dense(n_output_channels) )(o)
        m = Model(inputs=[i1,i2], outputs=[o])

        # Choose optimizer
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
          1e-3,
          decay_steps= 2000,
          decay_rate=1.,
          staircase=True)
        opt = optimizers.Adam( lr_schedule )

        # Put model together
        m.compile(optimizer= opt , loss=tf.keras.losses.MeanAbsoluteError(), metrics=[tf.keras.metrics.MeanSquaredError()])
        m.summary()
        plot_model(m  , to_file='model_plot.png' , show_shapes=True, show_layer_names=True)


        # Call-backs 
        early = tf.keras.callbacks.EarlyStopping(monitor = 'loss', min_delta = 1e-6, patience = 10, verbose = 1)
        st = time.time()
        m_his = m.fit( [ x_train_img_encoded.copy() , x_train_arry.copy() ] , y_train.copy(), epochs=EPOCH, validation_split=0.15, shuffle=True ,callbacks = [early], verbose = 1 , batch_size = 600 )
        train_time = time.time() - st

        All_train_time.append( train_time )

        kk = list(m_his.history.keys())
        np_loss_history = np.array( m_his.history[ kk[0] ] )
        np.savetxt('Repetition'+str(REP)+"/PCT-"+str(pct_count) +"loss.txt", np_loss_history, delimiter=",")

        np_v_loss_history = np.array( m_his.history[ kk[2] ] )
        np.savetxt('Repetition'+str(REP)+"/PCT-"+str(pct_count) +"val_loss.txt", np_v_loss_history, delimiter=",")

        np_metric_history = np.array( m_his.history[ kk[1] ] )
        np.savetxt('Repetition'+str(REP)+"/PCT-"+str(pct_count) +"metric.txt", np_metric_history, delimiter=",")

        np_val_metric_history = np.array( m_his.history[ kk[3] ] )
        np.savetxt('Repetition'+str(REP)+"/PCT-"+str(pct_count) +"val_metric.txt", np_val_metric_history, delimiter=",")


        # save model and architecture to single file
        try:
            m.save('Repetition'+str(REP)+"/PCT-"+str(pct_count) +"GRU_model.h5")
            print("Saved model to disk")
        except:
            pass

        print('Predicting')
        y_test_pred = m.predict( [x_test_img_encoded , x_test_arry] )
        y_test_pred2 = m.predict( [x_test2_img_encoded , x_test2_arry] )
        # y_test_pred_new = m.predict( [x_img_encoded_new , x_arry_new] )
        print('Done Predicting')
        model_evaluation = m.evaluate(x=[x_test_img_encoded , x_test_arry] , y=y_test_gt, batch_size= 100 )
        print(model_evaluation)
        print('\nOn a new strucutre:')
        model_evaluation = m.evaluate(x=[x_test2_img_encoded , x_test2_arry] , y=y_test_gt2, batch_size= 100 )
        print(model_evaluation)


        # print('\nOn a new geometry 1:')
        # model_evaluation = m.evaluate(x=[x_img_encoded_new[:50*50] , x_arry_new[:50*50]] , y=y_new[:50*50], batch_size= 100 )
        # print('On a new geometry 2:')
        # model_evaluation = m.evaluate(x=[x_img_encoded_new[50*50:100*50] , x_arry_new[50*50:100*50]] , y=y_new[50*50:100*50], batch_size= 100 )
        # print('On a new geometry 3:')
        # model_evaluation = m.evaluate(x=[x_img_encoded_new[100*50:] , x_arry_new[100*50:]] , y=y_new[100*50:], batch_size= 100 )



        #################################################  Save predictions  #######################################################################
        # # Apply inverse transform to inputs
        # for ss , curr_scaler in zip( x_scale_list , xScalers ):
        #     x_train_arry[:,:,ss] = curr_scaler.inverse_transform( x_train_arry[:,:,ss] )
        #     x_test_arry[:,:,ss] = curr_scaler.inverse_transform( x_test_arry[:,:,ss] )
        #     x_test2_arry[:,:,ss] = curr_scaler.inverse_transform( x_test2_arry[:,:,ss] )

        # Apply inverse transform to outputs
        for ss , curr_scaler in zip( range(n_output_channels) , yScalers ):
            # y_train[:,:,ss] = curr_scaler.inverse_transform( y_train[:,:,ss] )
            y_test_gt[:,:,ss] = curr_scaler.inverse_transform( y_test_gt[:,:,ss] )
            y_test_gt2[:,:,ss] = curr_scaler.inverse_transform( y_test_gt2[:,:,ss] )
            y_test_pred[:,:,ss] = curr_scaler.inverse_transform( y_test_pred[:,:,ss] )
            y_test_pred2[:,:,ss] = curr_scaler.inverse_transform( y_test_pred2[:,:,ss] )
            # y_new[:,:,ss] = curr_scaler.inverse_transform( y_new[:,:,ss] )
            # y_test_pred_new[:,:,ss] = curr_scaler.inverse_transform( y_test_pred_new[:,:,ss] )

        # Save data
        # f = open( 'X_data.npy' , 'wb' )
        # np.save( f , np.array([ x_train_img , x_test_img , x_test2_img , x_train_arry , x_test_arry , x_test2_arry ] ,dtype=object) )
        # f.close()
        f = open( 'Repetition'+str(REP)+"/PCT-"+str(pct_count) +'Y_data.npy' , 'wb' )
        np.save( f , np.array([ y_train , y_test_gt , y_test_gt2 ] ,dtype=object) )
        f.close()
        f = open( 'Repetition'+str(REP)+"/PCT-"+str(pct_count) +'Predictions.npy' , 'wb' )
        np.save( f , np.array([ y_test_pred , y_test_pred2 ] ,dtype=object) )
        f.close()



# Save all train times
f = open( 'TrainTime.npy' , 'wb' )
np.save( f , All_train_time )
f.close()
print( All_train_time )