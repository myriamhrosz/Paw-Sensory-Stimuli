# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 11:33:04 2021

@author: Myriam Hrosz
"""
# Import important libraries
import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.io as sio
import pandas as pd
import math as m
import scipy.stats as stats

path = (r'C:\Users\myria\Desktop\Purdue\Classes\BME 511\Final Project\538691R1L')
# Name files
csv_file = 'LIMB1_53869_1R1L_SPDLC_resnet50_691R1L_sp_limb1Nov12shuffle1_300000.csv'
fname = '538691R1L_000_002.mat'
# fall_file = 'Fall.mat'

# CSV file

# Determine Limb View
if csv_file[4]== '1':
    limb_view = 1
elif csv_file[4] == '2':
    limb_view = 2
else:
    assert str.isnumeric(csv_file[4]),"Make sure csv file starts with LIMB1 or LIMB2"

# Load Files
csv = pd.read_csv(path+'\\'+ csv_file,header=2).to_numpy()
file = sio.loadmat(path+'\\'+fname,squeeze_me= True,struct_as_record=False)
# Fall = sio.loadmat(path+'\\'+fall_file, squeeze_me=True,struct_as_record=False)
frames = file['info'].__dict__['frame']
stat = np.load(path+'//stat.npy',allow_pickle=True)
ops = np.load(path+'//ops.npy',allow_pickle=True).item()
iscell = np.load(path+'//iscell.npy',allow_pickle=True)
spks = np.load(path+'//spks.npy',allow_pickle=True)
F = np.load(path+'//F.npy',allow_pickle=True)
Fneu = np.load(path+'//Fneu.npy',allow_pickle=True)
# FILTER 1 (Linear Regression with Arbitrary Beta - 0.7)
# 1.Correct Neuropill with linear regression
Fs_1 = F - 0.7 * Fneu

#Plot F vs Fs
plt.figure(1)
plt.plot(F[0,:])
plt.plot(Fs_1[0,:])
plt.xlim(3000,4360)
plt.legend(['Raw','Without Neuropill'])
plt.title('Neural Response of First Recorded Neuron',fontsize=25)
plt.xlabel('Frames',fontsize=20)
plt.ylabel('Neural Response',fontsize=20)
# 2. keep only "iscell" w/ prob. > 0.75
iscellthresh = 0.75
iscellch = iscell[:,1] >= iscellthresh
Fs = Fs_1[iscellch,:]
Ncells = Fs.shape[0]

# 3. align neural data to first behavioral data frame
Npts_init = Fs.shape[1]
# maxpts = 5457; # based on weird ttl frame rate change, it shifts @ 5283)
Fn = Fs[:,frames[0]-1:Npts_init]
Fn_z = stats.zscore(Fn,axis=1)
Npts = Fn_z.shape[1]

# 4. Match every Neural Datapoint to a Behavioral Datapoint
# behavioral data (b) gave TTL outputs every 10 behavioral frames
# Behavioral frame rate is 30 Hz
b_ifi = 1/30 # behavioral inter-frame interval
n_fs = 7.81 # Neural frames /sec
nfs_inter = 1/n_fs # inter-frame interval for neural frames

video_pt = csv.shape[0]
b_pt = np.zeros((Npts,1))
x = 0
adjframes = frames - frames[1] + 1
for pt in range(video_pt-1):
    if pt % 4 == 1 or pt % 4 == 2:
        if b_pt[adjframes[x]] == 0:
            b_pt[adjframes[x]] = pt
        x+=1
b_pt = b_pt[b_pt!=0]
Fn = Fn_z[:,0:b_pt.shape[0]]
Npts = Fn.shape[1]

# B_pt to integer for later use
b_pt = b_pt.astype(int)

# Determine columns for front limbs (fl) and hind limbs (hl) (x,y,likelihood)
frontl = csv[:,1:4]
hindl = csv[:,4:7]
# 320 grain start and end (lg)  
lowg_start = csv[:,7:10]
lowg_end = csv[:,10:13]
# 80 grain start and end (hg)
highg_start = csv[:,13:16]
highg_end = csv[:,16:19]
# Mask with behavioral points
fl = frontl[b_pt,:]
hl = hindl[b_pt,:]
lg_start = lowg_start[b_pt,:]
lg_end = lowg_end[b_pt,:]
hg_start = highg_start[b_pt,:]
hg_end = highg_end[b_pt,:]

# Plot Behavior before and after maksing with neural data
plt.figure(2)
# Front Limb
plt.subplot(121)
plt.plot(frontl[:,0])
plt.plot(fl[:,0])
#plt.xlim(3000,4360)
plt.xlabel('Frames',fontsize=20)
plt.ylabel('X Position (pixels)',fontsize=20)
plt.legend(['Raw','Adjusted to Neural Data'])
plt.title("Front Limb X Coordinates",fontsize=25)
# Hind Limb
plt.subplot(122)
plt.plot(hindl[:,0])
plt.plot(hl[:,0])
#plt.xlim(3000,4360)
plt.xlabel('Frames',fontsize=20)
plt.ylabel('X Position (pixels)',fontsize=20)
plt.legend(['Raw','Adjusted to Neural Data'])
plt.title("Hind Limb X Coordinates",fontsize=25)
plt.show()

## FILTER
# Set threshold to indetify markers in frame
likelihood_thresh = 0.95;

#Plot front and hind limb position before and after threshold of likelihood
fig = plt.figure(3)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(fl[:,0])
ax2.plot(hl[:,0])
# var = ["fl","hl","lg_start","lg_end","hg_start","hg_end"];
mask_fl = fl[:,2] < likelihood_thresh
fl[mask_fl,0:1] = float("nan")
mask_hl = hl[:,2] < likelihood_thresh
hl[mask_hl,0:1] = float("nan")
mask_lgs = lg_start[:,2] < likelihood_thresh
lg_start[mask_lgs,0:1] = float("nan")
mask_lge = lg_end[:,2] < likelihood_thresh
lg_end[mask_lge,0:1] = float("nan")
mask_hgs = hg_start[:,2] < likelihood_thresh
hg_start[mask_hgs,0:1] = float("nan")
mask_hge = hg_end[:,2] < likelihood_thresh
hg_end[mask_hge,0:1] = float("nan")

# Idx: 1 = 320, 2 = 80, 3 = smooth wheel, 4 = disregard
# lg = 320, hg = 80
# add condition to keep status in last else but for debug 0.
Loc = np.empty((Npts,1))
Loc[:] = np.NaN

# Front Limb
ax1.plot(fl[:,0])
ax2.plot(hl[:,0])
ax1.set_xlabel('Frames',fontsize=20)
ax2.set_xlabel('Frames',fontsize=20)
ax1.set_ylabel('Position (pixels)',fontsize=20)
ax1.legend(['Before Threshold','After Threshold'])
ax1.set_title("Front Limb X Coordinates",fontsize=25)
ax2.legend(['Before Threshold','After Threshold'])
ax2.set_title("Hind Limb X Coordinates",fontsize=25)
plt.show()
# Determine starting point if no borders are showing
if m.isnan(lg_start[0,0]) and m.isnan(lg_end[0,0]) and m.isnan(hg_start[0,0]) and m.isnan(hg_end[0,0]):
    f = 3 # check future frame
    if hl[f,0]>=lg_start[f,0] or fl[f,0] <= lg_end[f,0]:
        status = 1
    elif fl[f,0]>=hg_start[f,0] or fl[f,0] <= hg_end[f,0]:
        status = 2
    elif fl[f,0]<=lg_start[f,0] or fl[f,0]<=hg_start[f,0] or hl[f,0]>=lg_end[f,0] or hl[f,0]>=hg_end[f,0]:
        status = 3
#Determine where the mouse is on the wheel only looking at xposition
for i in range(Npts):
    # On 320 Grit -- 1
    if(hl[i,0] >= lg_start[i,0]) or (fl[i,0] <= lg_end[i,0]):
        Loc[i] = 1
        status = 1
    # On 80 Grit -- 2
    elif(hl[i,0] >= hg_start[i,0]) or (fl[i,0] <= hg_end[i,0]) :
        Loc[i] = 2
        status = 2
    # On smooth wheel  -- 3
    elif(fl[i,0] <= lg_start[i,0]) or (hl[i,0] >= lg_end[i,0]) or (hl[i,0] >= hg_end[i,0]) or (fl[i,0] <= hg_start[i,0]):
        Loc[i] = 3
        status = 3
    # Half on 320 -- 4
    elif (hl[i,0] <= lg_end[i,0] and fl[i,0] > lg_end[i,0]) or (fl[i,0] >= lg_start[i,0] and hl[i,0] < lg_start[i,0]):
        Loc[i] = 4
        status = 4
    # Half on 80 -- 5
    elif (hl[i,0] <= hg_end[i,0] and fl[i,0] > hg_end[i,0]) or (fl[i,0] >= hg_start[i,0] and hl[i,0] < hg_start[i,0]):
        Loc[i] = 5
        status = 5
    else:
        Loc[i] = status

# 5. Now we have a behavioral timepoint associated with each 
# neural timepoint. Let's break the neural data into groups
# based either on hand-annotated data or on DLC data
# Loc = NaN(max(frames),0); # use original size since Megan annotated full movie
# # # Idx: 1 = 320, 2 = 80, 3 = wheel
# Now, based on the behavioral frame assigned to each neural frame, figure out location
b_vals = np.unique(Loc)
num_b = len(b_vals)
bCond = Loc

# Plot Beh Conditions as a function of neural frame
# plt.figure(4)
# plt.plot(bCond)
# plt.xlabel('Neural Frame #')
# plt.ylabel('Behavioral Condition')
# plt.title('1 = 320 Grit Sandpaper, 2 = 80 Grit Sandpaper, 3 = Plain Wheel, 4 = Half On 320, 5 = Half on 80')
# plt.show()
# Plot Behavior

## Make tuning curve to behavioral conditions
tune = np.empty((Ncells,num_b))
tune[:] = np.NaN

# Take the average of neural responses for each neuron
mask1 = bCond[:] == 1
mask2 = bCond[:] == 2
mask3 = bCond[:] == 3
mask4 = bCond[:] == 4
mask5 = bCond[:] == 5
for k in range(Ncells):
    tune[k,0] = np.mean(Fn[k,mask1[:,0]], axis = 0)
    tune[k,1] = np.mean(Fn[k,mask2[:,0]], axis = 0)
    tune[k,2] = np.mean(Fn[k,mask3[:,0]], axis = 0)
    tune[k,3] = np.mean(Fn[k,mask4[:,0]], axis = 0)
    tune[k,4] = np.mean(Fn[k,mask5[:,0]], axis = 0)
#Plot tuning curves for 100 neurons
plt.figure(6, figsize=(35,19))

for k in range(1,101):
    plt.subplot(10,10,k)
    plt.plot(tune[k-1,:])
    plt.title(k,fontsize=10)
    plt.xticks(np.arange(0,5),labels=['1','2','3','4','5'])
plt.show()

# # find maximum response
mxrspval = np.amax(tune,1)
mxrspidx = np.argmax(tune,1)
plt.figure(5,figsize=(20,12))
plt.hist(bCond,align='left')
plt.title('Amount of Frames on Each Texture',fontsize=25)
plt.ylabel('Neural Frames / # Cells',fontsize=20)
plt.xlabel('Behavioral Condition',fontsize=20)
plt.xticks(np.arange(1,6),labels=['320Grit','80Grit','Smooth Wheel','Half on 320 Grit','Half on 80 Grit'],fontsize=15)
plt.show()
plt.figure(7,figsize=(20,12))
plt.hist(mxrspidx,align='left')
plt.xlabel('Preferred Beh Condition',fontsize=20)
plt.ylabel('# cells',fontsize=20)
plt.title('Max Neural Response',fontsize=25)
plt.legend(['Frames Spent in Condition', 'Max Neural Resp'])
plt.xticks(np.arange(0,5),labels=['320','80','Smooth','alf 320','half 80'],fontsize=15)
plt.show()

#  Let's get fancy -- plot PD over the map of the neurons
#  1. read ops file that has mean image

im = np.empty((ops['Ly'], ops['Lx']))
im[:]=np.nan
for n in range(Ncells):
    ypix = stat[n]['ypix']
    xpix = stat[n]['xpix']
    im[ypix,xpix] = n+1
    
# # Plot the image of the brain
plt.figure(8, figsize=(20,15))
plt.imshow(ops['meanImg'],cmap='gray')
plt.imshow(im)
plt.legend()
plt.title('Raw Fluorescence',fontsize=25)
plt.show()

plt.figure(9,figsize=(20,15))
plt.imshow(ops['meanImg'], cmap='gray')
colorid = ['b','g','r','c','lime']
labels = ['320-grit','80-grit','Smooth Wheel','320-Half','80-Half']
for k in range(Ncells):
    xp = stat[k]['xpix']
    yp = stat[k]['ypix']
    plt.plot(xp,yp,color = colorid[mxrspidx[k]], linewidth=1.5)
 
import matplotlib.lines as mlines

h1 = mlines.Line2D([], [], color='blue',label=labels[0])
h2 = mlines.Line2D([], [], color='green', label=labels[1])
h3 = mlines.Line2D([], [], color='red', label=labels[2])
h4 = mlines.Line2D([], [], color='c', label=labels[3])
h5 = mlines.Line2D([], [], color='lime', label=labels[4])
plt.legend(handles=[h1,h2,h3,h4,h5],fontsize=15,loc="upper right")
plt.title('Cell Preferred Textures',fontsize=25)
plt.show()

plt.figure(10)
plt.plot(Fn[k,mask3[:,0]])
plt.plot(Fn[k,mask1[:,0]])
plt.plot(Fn[k,mask2[:,0]])
plt.xlim(3000,3576)
plt.legend(['Smooth Wheel','320 Grit','80 Grit'])
plt.title('Neural Response for different textures')

