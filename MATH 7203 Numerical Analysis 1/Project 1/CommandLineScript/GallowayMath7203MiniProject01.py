'''
Description: Functions for Water Level Detection Mini Project

Author: Josh Galloway
Version: 1.0
Date: 27 Feb 2020
'''

# # Canny Edge Detection Algoritm And Water Level Indication
# 
# ## Goal: Implement and Modify Canny Edge Detection Algorithm to Detect Water Level in A Vessel
# 
# 
# ### Outline of the Canny Edge Detection Algorithm
#     1. Noise Reduction
#     2. Intesity Gradient Calculation
#     3. Non-Maximum Supression
#     4. Double Thresholding
#     5. Hysteresis Thresholding
# 
# ### Modifications For Water Level Detection
#     Non-Maximum Supression Alteration
#     Adhoc Level Line Selection

'''Import Libraries'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv

'''Define function for the Sobel Operator'''
def sobel(A):
    '''Applies 3x3 Sobel Operator to Image Array and 
       returns Magnitude and Angle Arrays '''
    
    # Sobel Operator
    Gx = np.matrix([[-1,0,1],[-2,0,2],[-1,0,1]])
    Gy = Gx.T
    
    # Get Dimmensions of kernel and image
    kr,kc = Gx.shape
    r,c = A.shape
    
    # Dimension Output Arrays
    mag = np.zeros((r,c))
    angle = np.zeros((r,c))
    
    
    # Apply Sobel via Convolution
    o_r = kr//2  # offset for rows to center
    o_c = kc//2  # offset for columns to center
    for i in range(r - kr):
        for j in range(c - kc):
            # Perform X and Y Gradient Calcs on Slices of Image
            S1 = np.sum(np.multiply(Gx,A[i:i+kr,j:j+kc]))
            S2 = np.sum(np.multiply(Gy,A[i:i+kr,j:j+kc]))
            
            # Calculate Magnitude and Angle for each pixel
            mag[i+ o_r, j+o_c] = np.sqrt(S1*S1+S2*S2)
            angle[i+ o_r, j+o_c] = np.arctan2(S2,S1)

    return mag, angle

def nonMaxSupression(m,a):
    '''Round angle in radians to either 0°, 45°, 90°,
        or 135° and check neighbors in direction of gradient'''
    
    BP1 = 0.3927        # 22.5°
    BP2 = 1.1781        # 67.5°
    BP3 = 1.9635        # 112.5°
    BP4 = 2.7489        # 157.5°    

    s = m.copy()        # copy array 
    r,c = s.shape       # get rows and columns
    
    #loop through image skipping edge pixel
    for i in range(1,r-1):
        # i for rows
        for j in range(1,c-1):
            # j for cols
            
            a1 = np.abs(a[i,j])  # reflect to q 1&2        
            #  classify angles
            if (0<= a1 <BP1) or (BP4<= a1 <=np.pi):
                #closest to x-axis check east to west
                s[i,j] = s[i,j] if(s[i,j-1] < s[i,j] and s[i,j+1] < s[i,j]) else 0         
            elif (BP1<= a1 <BP2):
                # closest to 45° check SW and NE 
                s[i,j] = s[i,j] if(s[i+1,j-1] < s[i,j] and s[i-1,j+1] < s[i,j]) else 0 
            elif (BP2<= a1 <BP3):
                # closest to 90° North and South
                s[i,j] = s[i,j] if(s[i+1,j] < s[i,j] and s[i-1,j] < s[i,j]) else 0
            else:
                # closest to 135° NW and SE
                s[i,j] = s[i,j] if(s[i-1,j-1] < s[i,j] and s[i+1,j+1] < s[i,j]) else 0                
    return s


'''Double Threshold Function'''
def dblThreshold(A,H,L):
    '''Suppress Values in image, A, below L and highlight 
        pixel values above H.  Mark those in between as Weak'''
    STRONG = 255
    WEAK = 128
    
    # store shape of A
    r,c = A.shape
    B = np.zeros(r*c)  # create vector with same num pixels
    
    #flatten and cycle through image
    for i,pixel in enumerate(A.reshape(1,-1)[0,:]):
        if pixel > H:
            B[i] = STRONG
        if pixel < H and L < pixel:
            B[i] = WEAK
        # Array was init to zero so
        # no need to check for supression conditions
    return B.reshape(r,c)  


'''Define Hysteresis Thresholding Function'''
def hystTresh(A):
    '''Classify pixels as supressed or strong based on neighbors'''
    TOL = 25               # tolerance
    STRONG = 255           # Strong pixel
    WEAK = 128
    SUPRESSED = 0          # Supressed pixel
    WIN = 3                # Window Size of neighbors to check
    
    r,c = A.shape          # Get Shape of A
    B = A.copy()           # Copy A
    
    for i in range(1,r-1):
        # rows loop
        
        for j in range(1,c-1):
            # cols loop
            # check if pixel is weak
            if B[i,j] < WEAK + TOL and B[i,j] > WEAK - TOL:
                #is weak so check neighbors
                if np.max(B[i-1:i+WIN-1,j-1:j+WIN-1]) > STRONG - TOL:
                    B[i,j] = STRONG
                else:
                    B[i,j] = SUPRESSED 
    return B


'''Water Level Finding Funcitons'''

# altered non-maximum supression
def nonHorzSupression(m,a):
    '''Round angle in radians to either 0 or 90°,
    and check neighbors in direction of gradient'''
    
    BP1 = 0.3927        # 22.5°
    BP2 = 1.1781        # 67.5°
    BP3 = 1.9635        # 112.5°
    BP4 = 2.7489        # 157.5°    

    s = m.copy()        # copy array 
    r,c = s.shape       # get rows and columns
    
    #loop through image skipping edge pixel
    for i in range(1,r-1):
        # i for rows
        for j in range(1,c-1):
            # j for cols
            
            a1 = np.abs(a[i,j])  # reflect to q 1&2        
            #  classify angles
            if (BP2<= a1 <BP3):
                # closest to 90° North and South
                s[i,j] = s[i,j] if(s[i+1,j] < s[i,j] and s[i-1,j] < s[i,j]) else 0
            else:
                # not a horizontal line
                s[i,j] = 0                
    return s

# image windowing fucntion
def window(A,dim):
    '''Takes in image, x,y coordinates for start of window and 
       the height (H) and width (W) of the window.  dim = (x,y,H,W)
       Returns pixels in window'''
    x,y,H,W = dim # break out dimensions for clarity
    return A[y:y+H,x:x+W]


def normalDist(L):
    '''return an array of weights corresponding to the 
       length of L centered at L/2'''
    mu = L/2.
    sig = mu/3
    z = (np.linspace(0,L,L) - mu)/sig
    return 1/sig/np.sqrt(2*np.pi)*np.exp(-0.5*z*z)



'''Create a waterline detection function'''
def detectLevel(A,win_dim,thresh_HL,filtSK):
    # break out packed parameters for clarity
    X,Y,H,W = win_dim
    sigma,ksize = filtSK
    high,low = thresh_HL
    LW = 3
    
    # Window Images
    temp = window(A,win_dim) 
    # Stage 1 Filter
    temp = cv.GaussianBlur(temp,ksize,sigma,cv.BORDER_REPLICATE)
    #Stage 2 Gradient
    tempM,tempA = sobel(temp)
    # Stage 3 Non Max Supress (modified)
    temp = nonHorzSupression(tempM,tempA)
    # Stage 4 Double Thresh
    temp = dblThreshold(temp,high,low)
    # Stage 5 Hysteresis
    temp = hystTresh(temp)
    # Draw final line
    Water = np.argmax(np.multiply(np.sum(temp,axis=1),normalDist(H)))
    # Draw line across image
    A[Y+Water-LW//2:Y+Water+LW//2,:] = 0

    return A