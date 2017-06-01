import numpy as np
import os

path = '/jukebox/norman/knv/thesis/sherlock/fMRI_masked/'

mask_path = '/jukebox/norman/knv/thesis/sherlock/masks/binary/'
# dict from folder name to list of zero voxels
vox_to_del = np.load('zero_voxels.npz')
vox_to_del = vox_to_del['zvox']
vox_to_del = vox_to_del.item()


# delete rows from fMRI-masked 
for fol in os.listdir(path):
    if fol == '.DS_Store' or fol == 'README':
        continue
    zero_vox = np.array(vox_to_del[fol])
    print "Number of Voxels to Delete: " + str(len(zero_vox))
    for s in os.listdir(path + fol + '/'):
    	print "Folder: " + fol
    	print "Subject: " + s
    	data = np.load(path + fol + '/' + s)
    	data = data['data']
    	data = data.T
        print "Before: " + str(data.shape)
        data = np.delete(data, zero_vox, axis=0)
        print "After: " + str(data.shape)
        np.savez(path + fol + '/' + s, data=data)

# modify the masks themselves to NOT INCLUDE these voxels!
# i.e., set them to 0 instead of 1.
for mask in os.listdir(mask_path):
    if mask.endswith('.npz'):
        m = np.load(mask_path + mask)
        m = m['mask']
        print np.shape(m)
        print "Mask Name: " + mask
        maskname = mask.split(".")[0]
        zero_vox = np.array(vox_to_del[maskname])
        print "Number of Voxels to Zero Out: " + str(len(zero_vox))
        indices = np.nonzero(m)[0]
        if len(zero_vox) > 0:
            print "Before: " + str(m[indices[zero_vox]])
            m[indices[zero_vox]] = np.zeros(zero_vox.shape)
            print "After: " + str(m[indices[zero_vox]])
            np.savez(mask_path + mask, mask=m)

