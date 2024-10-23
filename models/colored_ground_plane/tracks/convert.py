"""
Script to color the background of test.png, which 
serves as albedo map for the ground plane (saved as x.png)
"""

import imageio
import numpy as np ;
import sys ;
from PIL import Image ;

import matplotlib.pyplot as plt ;

data = imageio.imread(sys.argv[1]) ;
d = np.array(data).astype("float32")


print(d.shape, d.max(), d[:,:,3].max())

reg_mask1 = np.zeros(d.shape[0:2], dtype=np.uint8).astype("bool") ;
reg_mask2 = np.zeros(d.shape[0:2], dtype=np.uint8).astype("bool") ;
reg_mask3 = np.zeros(d.shape[0:2], dtype=np.uint8).astype("bool") ;

reg_mask1[:,0:500] = True ;

reg_mask2[:,3400:4000] = True ;

reg_mask3[:,500:3400] = True ;

bg_mask = d[:,:,3] < 100 ;
fg_mask = d[:,:,3] > 100 ; 

d [:,:,0] = 0.0 ;
d [:,:,1] = 0.0 ;
d [:,:,2] = 0.0 ;
d [:,:,3] = 255.0 ;




# make straight track white
d[np.logical_and(fg_mask , reg_mask1)] = (0.,0.,0.,255.)
d[np.logical_and(bg_mask , reg_mask1)] = (255.,255.,255.,255.)

# curved track 1 red
d[np.logical_and(fg_mask, reg_mask2)] = (0.,0., 0.,255.)
d[np.logical_and(bg_mask, reg_mask2)] = (255.,0., 0.,255.)

# track 3 green
d[np.logical_and(fg_mask, reg_mask3)] = (0,0.,0,255)
d[np.logical_and(bg_mask, reg_mask3)] = (0,255.,0,255)


f,ax = plt.subplots(1,4) ;
ax[0].imshow(d[:,:,0])
ax[1].imshow(d[:,:,1])
ax[2].imshow(d[:,:,2])
ax[3].imshow(d[:,:,3])
plt.show()

# make zero3 green
print(d.shape)

imageio.imwrite("x.png",d.astype("uint8"))



