import numpy as np
import torch

# Simulating Synthetic Images
# The training images of two groups are simulated. Each image can contain 4 Gaussian distribution density functions. Let the 4 standard deviations be
#
# |  $\sigma_1$ | $\sigma_2$  |
#
# |  $\sigma_3$ | $\sigma_4$  |
#
# The 4 Gaussians are constructed such that
#
# 1. two diagonal Gaussians $\sigma_1,\sigma_4$ are linked to a factor of interest $mf$ (e.g. true effect between two classes)
# 2. two off-diagonal Gaussians $\sigma_2,\sigma_3$ are linked to two different confounding factors $cf_1, cf_2$. Currently, $\sigma_2$ is set to be empty so that there is only a single confounder. 
def generate_data(N, seed=4201):
    
    np.random.seed(seed)
    
    labels = np.zeros((N*2,))
    labels[N:] = 1

    # 2 confounding effects between 2 groups
    cf1 = np.zeros((N*2,))
    cf2 = np.zeros((N*2,))
    cf1[:N] = np.random.uniform(1,4,size=N) 
    cf1[N:] = np.random.uniform(3,6,size=N) 
    cf2[:N] = np.random.uniform(1,4,size=N) 
    cf2[N:] = np.random.uniform(3,6,size=N)

    # 2 major effects between 2 groups
    np.random.seed(seed+1)
    mf = np.zeros((N*2,))
    mf[:N] = np.random.uniform(1,4,size=N) 
    mf[N:] = np.random.uniform(3,6,size=N)
    
    # simulate images
    d = int(32)
    dh = d//2
    x = np.zeros((N*2,d,d,1)) 
    y = np.zeros((N*2,)) 
    y[N:] = 1
    for i in range(N*2):
        x[i,:dh,:dh,0] = gkern(kernlen=d//2, nsig=5)*mf[i]
        x[i,dh:,:dh,0] = gkern(kernlen=d//2, nsig=5)*cf1[i]
        x[i,dh:,dh:,0] = gkern(kernlen=d//2, nsig=5)*mf[i]
        x[i] = x[i] + np.random.normal(0,0.01,size=(d,d,1)) # random noise
        
    return labels, cf1, cf2, mf, x, y

def PIL2array(img):
    return numpy.array(img.getdata(),
                       numpy.uint8).reshape(img.size[1], img.size[0], 3)
def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = numpy.c_[arr, 255*numpy.ones((len(arr),1), numpy.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

# define accuracy function
def binary_acc(y_pred, y_true):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_results_sum/y_true.shape[0]
    acc = torch.round(acc * 100)
    return acc

def return_x(y,color,direction):
    
    if color == "blue" and direction == "up":
        m = (4.0 + 6.0/11.0)
        c = 0.5
    elif color == "blue" and direction == "down":
        m = -3.226
        c = 2.097
    elif color == "green" and direction == "up":
        m = 4.0
        c = -0.5
    elif color == "green" and direction == "down":
        m = -3.704
        c = 3.370
    elif color == "red" and direction == "up":
        m = 3.223
        c = -1.129
    elif color == "red" and direction == "down":
        m = -4.545
        c = 5.041
    else:
        m = 1
        c = 0
    
    return (y-c)/m

# x >= y
def big_equal(x,y):
    import numpy as np
    return x > y or np.allclose(x,y)

# x <= y
def less_equal(x,y):
    import numpy as np
    return x < y or np.allclose(x,y)

def convert_jet_to_grey(img_array,n):
    import numpy as np
    new_image = np.zeros((img_array.shape[0],img_array.shape[1]))
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            pixel_blue = img_array[i,j,2]
            pixel_green = img_array[i,j,1]
            pixel_red = img_array[i,j,0]
            if (pixel_blue < 1) and big_equal(pixel_blue,0.5) and less_equal(pixel_green,0.5) :
                #print "a1"
                #print "i,j = ",i,",",j
                new_image[i,j] = return_x(pixel_blue,"blue","up")**n
            elif np.allclose(pixel_blue,1.0) and big_equal(pixel_green,0):
                #print "b1"
                #print "i,j = ",i,",",j
                new_image[i,j] = return_x(pixel_green,"green","up")**n
            elif (pixel_blue < 1) and big_equal(pixel_blue,0.4) and big_equal(pixel_green,0.5):
                #print "c1"
                #print "i,j = ",i,",",j
                new_image[i,j] = return_x(pixel_green,"blue","down")**n
            elif (pixel_red < 1) and big_equal(pixel_red,0.4) and big_equal(pixel_green,0.5):
                #print "c2"
                #print "i,j = ",i,",",j
                new_image[i,j] = return_x(pixel_green,"red","up")**n
            elif np.allclose(pixel_red,1.0) and big_equal(pixel_green,0):
                #print "b2"
                #print "i,j = ",i,",",j
                new_image[i,j] = return_x(pixel_green,"green","down")**n
            elif (pixel_red < 1) and big_equal(pixel_red,0.5) and less_equal(pixel_green,0.5):
                #print "a2"
                #print "i,j = ",i,",",j
                new_image[i,j] = return_x(pixel_blue,"red","down")**n
                    
    return new_image

def gkern(kernlen=21, nsig=3):
    import numpy
    import scipy.stats as st
    
    """Returns a 2D Gaussian kernel array."""
    
    interval = (2*nsig+1.)/(kernlen)
    x = numpy.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = numpy.diff(st.norm.cdf(x))
    kernel_raw = numpy.sqrt(numpy.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel
