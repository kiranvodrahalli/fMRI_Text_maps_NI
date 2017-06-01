import numpy as np 
from numpy.linalg import norm
import nibabel as nib 
import os

fMRI_path = '/jukebox/hasson/janice/sherlock_movie_data/' #'/fastscratch/janice/sherlock_movie/smoothed_6mm/'
fmri_name_m = 'sherlock_movie_' # use this: this is the actual watching of the movie
fmri_name_r = 'sherlock_recall_' # this is subjects speaking about the movie
fmasked = 'fMRI_masked/'
subs = ['s' + str(i) for i in range(1, 18) if i != 5] # exclude subject 5 since missing some data
data_path = '/jukebox/norman/knv/thesis/sherlock/fMRI_masked/'
mask_path = '/jukebox/norman/knv/thesis/sherlock/masks/'
suf = '.nii' #'.nii.gz'
orig = 'orig/'
bi = 'binary/'


# get full brain data
def get_full_brain():
	save_path = data_path
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	all_subs = []
	for s in subs:
		print "Subject " + s
		fmri_file = fMRI_path + fmri_name_m + s + suf
		fd = nib.load(fmri_file)
		fd = fd.get_data()
		print np.shape(fd)
		x, y, z, T = np.shape(fd)
		fd = np.reshape(fd, (x*y*z, T))
		print "new shape: " + str(fd.shape)
		print "Saving " + save_path + fmri_name_m + s + '.npz'
		np.savez(save_path + fmri_name_m + s + '.npz', data=fd)
			
# apply all binary masks to the data
def apply_bin_masks():
	masks = mask_path + bi
	for m in os.listdir(masks):
		if m.endswith('.npz'):
			mask = np.load(masks + m)
			mask = mask['mask']
			print np.shape(mask)
			maskname = m.split(".")[0]
			print "Mask Name: " + maskname
			save_path = data_path + maskname + '/'
			print "Savepath: " + save_path
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			all_subs = []
			for s in subs:
				print "Subject " + s
				fmri_file = fMRI_path + fmri_name_m + s + suf
				fd = nib.load(fmri_file)
				fd = fd.get_data()
				print np.shape(fd)
				x, y, z, T = np.shape(fd)
				fd_new = []
				for t in range(T):
					vox_t = fd[:, :, :, t]
					vox_t = np.reshape(vox_t, x*y*z) #same reshaping as mask
					vox_t = vox_t[np.nonzero(mask)[0]]
					fd_new.append(vox_t)
				masked = np.matrix(fd_new)
				#fd = np.reshape(fd, (T, x*y*z))
				#print "fd shape"
				#print np.shape(fd)
				#print norm(fd)
				#masked = fd[:, np.nonzero(mask)[0]]
				#print "Now masked"
				#print np.shape(masked)
				#print norm(masked)
				#all_subs.append(masked)
				print "Saving " + save_path + fmri_name_m + s + '.npz'
				np.savez(save_path + fmri_name_m + s + '.npz', data=masked)
			#all_subs = np.matrix(all_subs)
			#print "Saving overall file with all subs: " + save_path + fmri_name_m + 'all_subs' + '.npz'
			#np.savez(save_path + fmri_name_m + 'all_subs.npz', data=all_subs)


# create binary masks
def create_bin_masks():
	i_files = mask_path + orig
	output = mask_path + bi
	for i in os.listdir(i_files):
		# exclude the full brain mask; just shows you what parts of the brain each of the masks are
		if i.endswith('.nii') and (i != 'MNI152_T1_3mm_brain.nii'):
			curr_f = i_files + str(i)
			print "Current File: " + curr_f
			f = nib.load(curr_f)
			data = f.get_data()
			bmask = binarize(data)
			bmask = np.array(bmask)
			print np.shape(bmask)
			out_f = output + str(i)
			np.savez(out_f + '.npz', mask=bmask)

# number non-zero
def nnz(mask):
	I, J, K = np.shape(mask)
	mask = np.reshape(mask, I*J*K)
	return np.shape(mask[np.nonzero(mask)])

# convert mask into binary
def binarize(mask):
	I, J, K = np.shape(mask)
	mask = np.reshape(mask, I*J*K)
	nnz = np.nonzero(mask)
	mask[nnz] = np.ones(np.shape(nnz))
	return mask


#apply_bin_masks()
if __name__=='__main__':
	get_full_brain()
