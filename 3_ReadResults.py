import numpy as np
import matplotlib.pyplot as plt

try:
	# Read training results
	loss = np.loadtxt('./Repetition0/PCT-1loss.txt')
	metric = np.loadtxt('./Repetition0/PCT-1metric.txt')

	val_loss = np.loadtxt('./Repetition0/PCT-1val_loss.txt')
	val_metric = np.loadtxt('./Repetition0/PCT-1val_metric.txt')

	plt.figure()
	plt.plot( loss , 'k' )
	plt.plot( val_loss , 'r--' )
	plt.xlabel('Training epoch')
	plt.ylabel('Loss')
	plt.legend(['Training set','Validation set'])

	plt.figure()
	plt.plot( metric , 'k' )
	plt.plot( val_metric , 'r--' )
	plt.xlabel('Training epoch')
	plt.ylabel('Metric')
	plt.legend(['Training set','Validation set'])
except:
	pass


# Plot stress strain curves
f=open('./Repetition0/PCT-1Y_data.npy','rb')
y_train , y_test_gt , y_test_gt2 = np.load(f,allow_pickle=True)
f.close()
f=open('./Repetition0/PCT-1Predictions.npy','rb')
y_test_pred , y_test_pred2 = np.load(f,allow_pickle=True)
f.close()




#####################################################################################################################
# Plot by percentile
y_vec = y_test_gt.copy()
y_pred = y_test_pred.copy()
mae = np.mean( np.abs( y_vec - y_pred ) , axis = 1 )
y_labels = ['Reaction force [mN]','Plastic dissipation [mJ]','Damage dissipation [mJ]','Elastic strain energy [mJ]']

p = 0
for i in range(4):
	curr_mae = mae[:,i]
	idx = np.arange(len(curr_mae))
	curr_mae, idx = zip(*sorted(zip(curr_mae, idx)))

	for pct in [ 0.25 , 0.5 , 0.75 , 1. ]:
		ax = plt.subplot(4, 4, p+1)
		p += 1

		if pct == 0.:
			ii = 0
		elif pct == 1.:
			ii = -1
		else:
			ii = int(round(pct*len(curr_mae)))
		plt.plot( y_vec[ idx[ii],:,i] , 'tab:orange' )
		plt.plot( y_pred[idx[ii],:,i] , 'tab:purple' )
		if i == 3:
			plt.xlabel('Time steps', fontsize=11)
		if pct == 0.25:
			plt.ylabel(y_labels[i], fontsize=11)

		if i == 0:
			if pct == 0.25:
				plt.title('25th percentile', fontsize=16)
			if pct == 0.5:
				plt.title('50th percentile', fontsize=16)
			if pct == 0.75:
				plt.title('75th percentile', fontsize=16)
			if pct == 1.:
				plt.title('Worst')
		plt.legend(['Ground truth','Pred, MAE='+"{0:.2g}".format(curr_mae[ii])], fontsize=9)

# plt.tight_layout()
plt.show()