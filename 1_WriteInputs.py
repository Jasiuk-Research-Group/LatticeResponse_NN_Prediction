import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, skeletonize , dilation
from skimage.measure import label
from skimage.transform import resize
from skimage.io import imread
import pickle 

def ToImg( key , NN , nc ):
	# Fixerd parameters
	center = [ 232 , 363 ]
	N = 465
	dx = 20. / 444

	img = imread('./Design cross sections/K' + key + '.png')
	# Crop image
	img = img[ center[0]-N//2:center[0]+N//2 , center[1]-N//2:center[1]+N//2 ]

	# Binarize
	img = label(img < 1)

	# Remove disconnected parts
	lbl = label( img )
	for ll in range(1,20):
		mask = (lbl == ll)
		if np.sum( mask ) > 20:
			break
	img = mask

	# Resize image
	c = np.linspace( 0. , 1. , 464 )
	X , Y = np.meshgrid( c , c )
	X = X[img]
	Y = Y[img]

	c = np.linspace( 0. , 1. , NN )
	dc = c[1] - c[0]
	img_c = np.zeros( [NN,NN] )
	for x , y in zip( X , Y ):
		i = int(round(y/dc))
		j = int(round(x/dc))
		img_c[ i , j ] = 1

	# Get skeleton
	img = skeletonize( img_c )
	return img


N_samples = 15000
NN = 50
nc = 30
repeat = 12 # Number of times to randomly sample a set of input data

# Read parameters
f = open( './AllParameters.npy' , 'rb' )
All_Keys , Thickness , Strain_rate = np.load( f )
f.close()

# Read energy series
f = open( './FE_Results/EnergySeries.npy' , 'rb' )
PD , DMD , ESE = np.load(f)
f.close()

Inputs1 = np.zeros( [ N_samples * repeat , 128 , 128 ] )
Inputs2 = np.zeros( [ N_samples * repeat , NN , 6 ] )
Outputs = np.zeros( [ N_samples * repeat , NN , 4 ] )
Input_keys = []


# Begin writing inputs
E = 109778.e6 #Pa
rho = 4428. # kg/m^3
c0 = np.sqrt( E / rho ) # m/s
ey = 0.2 / 100.
L = 10.
t_elastic = ( L / 1000. ) / c0 # s

img_dict = {}
for uk in np.unique( All_Keys[:N_samples] ):
	img_dict[ uk ] = ToImg( uk , 128 , nc )
print('Done saving all images')

with open('ImgDict.pkl', 'wb') as f:
	pickle.dump(img_dict, f)


print('Begin writing...')
strain_array = np.linspace( 0. , 0.2 , 201 )
for i in range( N_samples ):
	print('Writing ' + str(i))

	my_key = str(All_Keys[i])
	my_thickness = float(Thickness[i])
	my_strain_rate = float(Strain_rate[i])

	# Output: Interpolated force-displacement curve
	f = open( './FE_Results/Sim-' + str(i) +'.npy' , 'rb' )
	F,U = np.load(f)
	F = np.abs(F)
	U = np.abs(U)
	f.close()

	final_strains = np.random.rand( repeat ) * 0.15 + 0.05 #  in range [ 0.05 , 0.2 ]
	final_strains[0] = 0.2
	for rr in range( repeat ):
		# Key
		Input_keys.append( my_key )

		# Thickness
		Inputs2[ i * repeat + rr , : , 0 ] = my_thickness

		# log10( Strain rate )
		Inputs2[ i * repeat + rr , : , 1 ] = np.log10( my_strain_rate )

		# Final strain
		final_strain = final_strains[rr]
		Inputs2[ i * repeat + rr , : , 2 ] = final_strain

		# Time at each output point
		step_time = final_strain / my_strain_rate
		time_at_each_pt = np.linspace( 0. , step_time , NN )
		Inputs2[ i * repeat + rr , : , 3 ] = time_at_each_pt.copy()

		# Strain at each output point
		strain_at_each_pt = np.linspace( 0. , final_strain , NN )
		Inputs2[ i * repeat + rr , : , 4 ] = strain_at_each_pt.copy()

		# Stress flag
		stress_flag = np.ones( NN )
		stress_flag[ time_at_each_pt < t_elastic ] = 0
		Inputs2[ i * repeat + rr , : , 5 ] = stress_flag.copy()

		# Outputs
		eout = np.linspace( 0. , final_strain , NN )
		Outputs[ i * repeat + rr , : , 0 ] = np.interp( np.linspace( 0. , U[-1]/0.2*final_strain , NN ) , U , F )
		Outputs[ i * repeat + rr , : , 1 ] = np.interp( eout , strain_array , PD[i,:] )
		Outputs[ i * repeat + rr , : , 2 ] = np.interp( eout , strain_array , DMD[i,:] )
		Outputs[ i * repeat + rr , : , 3 ] = np.interp( eout , strain_array , ESE[i,:] )


f = open( 'Inputs2.npy' , 'wb' )
np.save( f , Inputs2 )
f.close()
f = open( 'InputKeys.npy' , 'wb' )
np.save( f , Input_keys )
f.close()
f = open( 'Outputs.npy' , 'wb' )
np.save( f , Outputs )
f.close()