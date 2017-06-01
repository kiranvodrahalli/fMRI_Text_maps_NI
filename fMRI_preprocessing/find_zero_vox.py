import numpy as np
import os

path = '/jukebox/norman/knv/thesis/sherlock/fMRI_masked/'

# dict from folder name to list of zero voxels
vox_to_del = dict()

for fol in os.listdir(path):
    if fol == '.DS_Store' or fol == 'README':
        continue
    zero_vox = set()
    for s in os.listdir(path + fol + '/'):
    	print "Folder: " + fol
    	print "Subject: " + s
    	data = np.load(path + fol + '/' + s)
    	data = data['data']
    	data = data.T
    	print data.shape
    	zeros = np.where(~data.any(axis=1))[0]
    	print data[zeros, :].shape
    	for z in zeros:
    		zero_vox.add(z)
    new_vox = []
    for z in zero_vox:
    	new_vox.append(z)
    vox_to_del[fol] = new_vox
    print "Zero Voxels for Folder " + fol + ": " + str(new_vox)

np.savez('zero_voxels.npz', zvox=vox_to_del)