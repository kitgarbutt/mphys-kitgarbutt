import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from tqdm.notebook import tqdm_notebook
import PIL
from PIL import Image


# -------- Assembly Parameters -------- #

led_position = (0.875,0.05) # Position (x,y) of LED from origin
x_led = led_position[0]
y_led = led_position[1]
alpha_x = 0.8
alpha_y = 0.0000075
R = 1
x_baffle = alpha_x*R
y_baffle = alpha_y*R
camera_position = 0.95 # Camera y-position
h_c = 1-camera_position
leaf_width = 0.001
N = 100000 # Number of photons included in the interaction
spec_prob = 0.01
abs_spec = 0.01
lens_upper = np.sqrt(1-(camera_position**2))
diffuser_limit_l = 0.76
diffuser_limit_u = 0.99
diffuser_r = 0.115
f = 0.3
u = camera_position
v = 1/((1/f)-(1/u))
ccd_height = u + v
mag = v/u
ccd_upper = mag*lens_upper
ccd_lower = mag*(-lens_upper)


# -------- LED Emission Functions -------- #

def lambert(theta):
    mn=0          # Lowest value of domain
    mx=math.pi    # Highest value of domain
    bound=1       # Upper bound of PDF value    
    
    while True:
        x=random.uniform(mn,mx)      # Choose an x value inside the desired sampling domain.
        y=random.uniform(0,bound)    # Choose a x value between 0 and the maximum PDF value.
        pdf=math.sin(x)              # Calculate PDF
        if y<pdf:                    # Does (x,y) fall in the PDF?
            """
            if (x + (math.pi/2) - theta) < 0:
                return x
            elif (x + (math.pi/2) - theta) > math.pi:
                return x
            else:
                return x + (math.pi/2) - theta
            """
            return x

def led():
    mn=0          # Lowest value of domain
    mx=math.pi    # Highest value of domain
    bound=1       # Upper bound of PDF value
    diffuser_x = random.uniform(diffuser_limit_l,diffuser_limit_u)
    pos = diffuser_x,y_led
    while True:
        x=random.uniform(mn,mx)      # Choose an x value inside the desired sampling domain.
        y=random.uniform(0,bound)    # Choose a x value between 0 and the maximum PDF value.
        pdf=math.sin(x)              # Calculate PDF
        if y<pdf:                    # Does (x,y) fall in the PDF?            
            dir = math.cos(x), math.sin(x) #Gets emission vector from lambertian distribution                       
            return (pos,dir)
        
def diffuser():
    mn=0          # Lowest value of domain
    mx=math.pi    # Highest value of domain
    bound=1       # Upper bound of PDF value
    while True:
        x=random.uniform(mn,mx)      # Choose an x value inside the desired sampling domain.
        y=random.uniform(0,bound)    # Choose a x value between 0 and the maximum PDF value.
        pdf=math.sin(x)              # Calculate PDF
        if y<pdf:                    # Does (x,y) fall in the PDF? 
            theta1 = y
            alpha = (math.pi/2) - theta1
            x_diff = x_led + (diffuser_r*math.cos(theta1))
            y_diff = (diffuser_r*math.sin(theta1))
            while True:
                xx=random.uniform(mn,mx)      # Choose an x value inside the desired sampling domain.
                yy=random.uniform(0,bound)    # Choose a x value between 0 and the maximum PDF value.
                pdf=math.sin(xx)              # Calculate PDF
                if yy<pdf:                    # Does (x,y) fall in the PDF? 
                    theta2 = yy - alpha
                    if theta2 >= math.pi:
                        theta3 = math.pi-theta2
                    if theta2 <= 0:
                        theta3 = 0-theta2
                    else:
                        theta3 = theta2
                    pos = x_diff, y_diff
                    dir = math.cos(theta3), math.sin(theta3) #Gets emission vector from lambertian distribution                       
                    return (pos,dir)

# -------- First Interaction -------- #
 
def first_interaction_surface(pos,dir):
    theta = np.arctan2(dir[1],dir[0])                                              # Marks each interaction type (0 = vertical, 1 = camera, 2 = circle, 3 = base, 5 = baffle)                                   
    #a = (R-y_led)-h_c                                                             # Distance in the y-direction from LED to camera
    theta_baffle = math.pi - np.arctan2((y_baffle - pos[1]),(pos[0] - x_baffle))     # Angle from -x direction to top edge of baffle, measured from LED
    theta_vertical = math.pi - np.arctan2((camera_position-pos[1]),pos[0])           # Angle from -x direction to vertical-camera intersection, measured from LED.
    theta_camera = math.pi - np.arctan2((camera_position-pos[1]),(pos[0]-lens_upper))# Angle from -x direction to camera-circle intersection, measured from LED.
    if 0 < theta and theta < theta_camera:
        surface_counter = 2
    elif theta_camera < theta and theta < theta_vertical:
        surface_counter = 1
    elif theta_vertical < theta and theta < theta_baffle:
        surface_counter = 0
    else:
        surface_counter = 5    
    return(surface_counter)                                                                       

        
def horizontal_specular(pos,dir):
    # Gives new trajectory of photon that is specularly reflected on the horizontal surface
    new_dir = dir[0],-dir[1]
    return(pos,new_dir)    

def on_or_off_leaf(pos,dir):
    # Determines whether photon interacts with the leaf or the baseline
    if 0 < pos[0] < leaf_width:
        return 1
    if pos[0] > leaf_width:
        return 0

def reflect_or_absorb(pos,dir):
    # Accounts for reflectivity of leaf
    if on_or_off_leaf(pos,dir) == 1:
        rand = random.random()
        if rand > 0.05: #Absorbed
            return 1
        else:           #Reflected
            return 0
    else:
        return 0
        
def base_interaction(pos,dir):                     # Determines whether a photon hitting the x axis will be absorbed by the 
    if reflect_or_absorb(pos,dir) == 1:            # leaf or reflected, and also records x coordinate of absorption
        x = np.arctan2(dir[1],dir[0]) + math.pi
        return(12345, pos[0], x)               # Returns all output with a dummy element to distinguish from relection
    else:
        theta = np.arctan2(dir[1],dir[0]) + math.pi
        x = lambert(theta)
        dir = np.cos(x), np.sin(x)
        return(pos,dir)
    
def circle_interaction(pos,dir):          # Function used to define the new position and direction of the photon after interaction with the 
    y = np.random.uniform(0,1)
    z = np.random.uniform(0,1)            # Random number         
    op = math.atan(pos[1]/pos[0])         # Angle between origin and point on circle
    theta_dir = np.arctan2(dir[1],dir[0]) # Direction angle of photon
    theta_inc = theta_dir-op
    if z < abs_spec:
        return(12345,pos)
    else:
        if y < spec_prob:                        
            if theta_dir > op:                        # If the photon comes from below the normal, the output angle is above the normal
                alpha = np.pi + 4*op - 3*theta_dir
                dir = math.cos(alpha), math.sin(alpha)
                return(pos,dir)
            else:                                     # And vice versa
                alpha = 90 + op - theta_dir
                dir = math.cos(alpha), math.sin(alpha)
                return(pos,dir)
        else: 
            if theta_dir > op:
                theta_inc = math.pi/2 + theta_dir - op
                alpha =  op + math.pi/2 + lambert(theta_inc) # New angle of trajectory
                dir = math.cos(alpha), math.sin(alpha)       # Direction vector from angle
                return(pos,dir)
            else:
                theta_inc = math.pi/2 - theta_dir + op
                alpha =  op + math.pi/2 + lambert(theta_inc) # New angle of trajectory
                dir = math.cos(alpha), math.sin(alpha)       # Direction vector from angle
                return(pos,dir)
    
def vertical_interaction(pos,dir):
    # Function used to define the new direction of the photon after interaction with the vertical
    dir = dir[0]*(-1), dir[1]
    return(pos,dir)
    
def baffler_interaction(pos,dir):
    y = np.random.uniform(0,1)
    z = np.random.uniform(0,1)           # Random number         
    theta_dir = np.arctan2(dir[1],dir[0])# Direction angle of photon
    if z < abs_spec:
        return(12345,pos)
    else:
        if y < spec_prob:
            dir = dir[0]*(-1),dir[1]
            return(pos,dir)
        else:
            if math.pi*(-1/2) < theta_dir < math.pi*(1/2):
                beta = theta_dir + (math.pi/2)
                x = lambert(beta)
                alpha = x + math.pi*(1/2)
                dir = math.cos(alpha), math.sin(alpha) # Direction vector from angle
                return(pos,dir)
            else:
                beta = theta_dir + (math.pi/2)
                x = lambert(beta)
                alpha = x - math.pi*(1/2)
                dir = math.cos(alpha), math.sin(alpha) # Direction vector from angle
                return(pos,dir)
            


def camera_interaction(pos,dir):      # Function used to define the new position and direction of the photon after interaction with the camera
    x = np.arctan2(dir[1],dir[0])
    lens_reflectance = 0.002
    y = np.random.uniform(0,1)
    if y < lens_reflectance:          # Small chance of reflectance
        dir = dir[0], dir[1]*(-1)
        return(pos,dir)
    else:
        return(12345, pos[0], x)        # Absorption output
    
def intersect_circle(pos,dir): 
    m = dir[1]/dir[0]                        # slope gradient
    a = m**2 + 1                             # Quadratic formula parameters for line-curve intersect
    b = 2*pos[1]*m - 2*pos[0]*(m**2)
    c = (pos[0]**2)*(m**2) + pos[1]**2 -2*pos[0]*pos[1]*m - 1
    x1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)  # Calculates x values of intersect points from quadratic formula
    x2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)  #
    y1 = np.sqrt(1-x1**2)                    # Calculates y values of intersect points from quadratic formula
    y2 = np.sqrt(1-x2**2)                    #
    if x2 < 0 or y2 < 0 or x2 > 1 or y2 > 1:
        return(x1,y1)
    elif x1 < 0 or y1 < 0 or x1 > 1 or y1 > 1:
        return(x2,y2)
    else:
        return(x1,y1,x2,y2)

def intersect_base(pos,dir):
    m = dir[1]/dir[0]         # slope gradient
    c = pos[1] - m*pos[0]     # y intercept
    x = -c/m
    return(x,0)

def intersect_vertical(pos, dir):
    m = dir[1]/dir[0]             # slope gradient
    c = pos[1] - m*pos[0]         # y intercept
    y = c
    return(0,y)

def intersect_camera(pos,dir):
    m = dir[1]/dir[0]             # slope gradient
    c = pos[1] - m*pos[0]         # y intercept
    x = (camera_position-c)/m     # x coordinate of intersection with camera
    return(x,camera_position)

def intersect_baffler(pos,dir):
    m = dir[1]/dir[0]             # slope gradient
    c = pos[1] - m*pos[0] 
    y = (m*x_baffle) + c
    return(x_baffle,y)

# -------- Subsequent Interactions -------- #
def interaction_surface(pos, dir):
    vertical = intersect_vertical(pos,dir)  
    baseline = intersect_base(pos,dir)      # assigns each coordinate of interaction a variable.
    camera = intersect_camera(pos,dir)      
    circle = intersect_circle(pos,dir) 
    baffler = intersect_baffler(pos,dir)
    if len(circle) > 2:
        circle1 = circle[0],circle[1]       
        circle2 = circle[2],circle[3]      # in the case where the line crosses two positive points on the circle, 
    else:                                  # this splits the variable into 2. otherwise it makes one automatically invalid.
        circle1 = circle                   
        circle2 = -2,-2                     
    which = []     # creates empty array
    if 0 < vertical[1] <= camera_position:                                       
        which = np.append(which, (0,vertical[1]))                   
    if 0 < baseline[0] <= 1:                                       
        which = np.append(which, (baseline[0],0))             # This section tests whether each point lies within the 
    if 0 < camera[0] <= lens_upper:                           # boundaries of the geometry, and passes them based on this.
        which = np.append(which, (camera[0],camera_position))          
    if lens_upper < circle1[0] <= 1 and 0 < circle1[1] <= camera_position:                  
        which = np.append(which, (circle1[0],circle1[1]))            
    if lens_upper < circle2[0] <= 1 and 0 < circle2[1] <= camera_position:                 
        which = np.append(which, (circle2[0],circle2[1]))
    if 0 < baffler[1] <= y_baffle:
        which = np.append(which, (baffler[0],baffler[1]))
    pos1 = which[0],which[1]                    # Splits into two points (always going to be two points that are valid)
    if len(which) == 2:
        return(pos1)
    else:
        if len(which) == 4:
            pos2 = which[2],which[3]
            pos3 = -3,-3
        else:
            pos2 = which[2],which[3]
            pos3 = which[4],which[5]
        delta1 = np.sqrt((pos[1]-pos1[1])**2 + (pos[0]-pos1[0])**2)   # For each point, determines the distance from initial position
        delta2 = np.sqrt((pos[1]-pos2[1])**2 + (pos[0]-pos2[0])**2)
        delta3 = np.sqrt((pos[1]-pos3[1])**2 + (pos[0]-pos2[0])**2)
        if delta1 == 0:
            if delta2 > delta3:
                return(pos3)
            else:
                return(pos2)
            
        if delta2 == 0:
            if delta1 > delta3:
                return(pos3)
            else:
                return(pos1)
        if delta3 == 0:
            if delta1 > delta2:
                return(pos2)
            else:
                return(pos1)
        else:                   # If no points are equal to current position, must be LED
            if pos1[1] == 0:    # Exclude base interactions
                if delta2 > delta3:
                    return(pos3)
                else:
                    return(pos2)
            if pos2[1] == 0:
                if delta1 > delta3:
                    return(pos3)
                else:
                    return(pos1)
            else:
                if delta1 > delta2:
                    return(pos2)
                else:
                    return(pos1)

def lens_transformation(theta_i, x_i):
    if 0 < theta_i < (math.pi/2):
        uprime = x_i*math.tan(theta_i)
        vprime = 1/((1/f)-(1/uprime))
        theta_f = (math.pi/2) + np.arctan2(x_i,vprime)
    elif (math.pi/2) < theta_i < math.pi:
        uprime = (-x_i)*math.tan(math.pi-theta_i)
        vprime = 1/((1/f)-(1/uprime))
        theta_f = math.pi - np.arctan2(vprime,x_i)
    elif theta_i == math.pi/2:
        theta_f = (math.pi/2) + np.arctan2(x_i,f)
    else: 
        return 10
    m = math.sin(theta_f)/math.cos(theta_f)
    c = camera_position - (m*x_i)
    x_f = (ccd_height-c)/m
    """
    registered = 1
    while registered == 1:
        if ccd_lower <= x_f <= ccd_upper:
            registered = 0
        elif x_f > ccd_upper:
            x_f = ccd_upper - x_f
        elif x_f < ccd_lower:
            x_f =  ccd_lower - x_f
    """
    return x_f
        
        
    
## -------- SCRIPT -------- ##
counter = 0
position_record = []
direction_record = []
RT_surface = []
camera_pos = []
camera_direction = []
logx=[]
logy=[]
ccd_position =[]
pbar = tqdm_notebook(total = N)
while counter < N:

    pos,dir = led()
    #RT_surface = np.append(RT_surface,4)                  # Defining the emission position and direction of the photon
    #logx = np.append(logx,pos[0])
    #logy = np.append(logy,pos[1])
    absorbed = 0
    surface_counter = first_interaction_surface(pos,dir)  # Determining the first interaction surface of the photon, recording it as a number

    if surface_counter == 0:                              # If first interaction is with vertical, call relevant intersection and interaction                                                         
        pos1 = intersect_vertical(pos,dir)                #functions
        pos,dir = vertical_interaction(pos1,dir)
        #RT_surface = np.append(RT_surface,0)                  # Defining the emission position and direction of the photon
        #logx = np.append(logx,pos[0])
        #logy = np.append(logy,pos[1])
        
    elif surface_counter == 1:                            # If first interaction is with camera module, call relevant intersection and                                                           
        pos1 = intersect_camera(pos,dir)                  #interaction functions
        ci = camera_interaction(pos1,dir)
        #RT_surface = np.append(RT_surface,1)                  # Defining the emission position and direction of the photon
        #logx = np.append(logx,pos[0])
        #logy = np.append(logy,pos[1])
        if ci[0] == 12345:
                absorbed =+ 1
                breaker = 1
        else:
            pos,dir = ci[0],ci[1]
        
    elif surface_counter == 2:                            # If first interaction is with circle, call relevant intersection and interaction                                                          #functions
        pos1 = intersect_circle(pos,dir)
        cii = circle_interaction(pos1,dir)
        #RT_surface = np.append(RT_surface,2)                  # Defining the emission position and direction of the photon
        #logx = np.append(logx,pos[0])
        #logy = np.append(logy,pos[1])
        if cii[0] == 12345:
            absorbed =+ 1
            breaker = 2                
        else:
            pos,dir = cii[0],cii[1]
        
    elif surface_counter == 5:                            # If first interaction is with circle, call relevant intersection and interaction                                                          #functions
        pos1 = intersect_baffler(pos,dir)
        bfi = baffler_interaction(pos1,dir)
        #RT_surface = np.append(RT_surface,5)                  # Defining the emission position and direction of the photon
        #logx = np.append(logx,pos[0])
        #logy = np.append(logy,pos[1]) 
        if bfi[0] == 12345:
            absorbed =+ 1
            breaker = 3                
        else:
            pos,dir = bfi[0],bfi[1]

    
    while absorbed == 0:
        breaker = 2
        pos = interaction_surface(pos,dir)              # Redefine pos as new position
        if pos[0] == 0:
            vi = vertical_interaction(pos,dir)
            #logx = np.append(logx,pos[0])        #
            #logy = np.append(logy,pos[1])        #
            pos, dir = vi[0],vi[1]
            #RT_surface = np.append(RT_surface,0) #
            continue                                      # If x = 0, vertical function used for new direction, redefines dir
        if pos[1] == 0:
            bi = base_interaction(pos,dir)    # If y = 0, horizontal function used for new direction, redefines dir
            #logx = np.append(logx,pos[0])           #
            #logy = np.append(logy,pos[1])           #
            #RT_surface = np.append(RT_surface,3)    #
            if bi[0] == 12345:                            # If output is leaf_record, photon absorbed and loop stopped
                absorbed =+ 1
                breaker = 0
            else:
                pos,dir = bi[0],bi[1]
                
                continue                                    # Or new direction, redefines dir
        if pos[1] == camera_position:
            ci = camera_interaction(pos,dir)
            #logx = np.append(logx,pos[0])           #
            #logy = np.append(logy,pos[1])           #
            #RT_surface = np.append(RT_surface,1)    #
            if ci[0] == 12345:
                absorbed =+ 1
                breaker = 1
            else:
                pos,dir = ci[0],ci[1]
                continue
        if pos[0] == x_baffle:
            bfi = baffler_interaction(pos,dir)
            #logx = np.append(logx,pos[0])            #
            #logy = np.append(logy,pos[1])            #
            #RT_surface = np.append(RT_surface,5)     #
            if bfi[0] == 12345:
                absorbed =+ 1
                breaker = 3                
            else:
                pos,dir = bfi[0],bfi[1]
                continue
        else:
            cii = circle_interaction(pos,dir)
            #logx = np.append(logx,pos[0])           #
            #logy = np.append(logy,pos[1])           #
            #RT_surface = np.append(RT_surface,2)    #
            if cii[0] == 12345:
                absorbed =+ 1
                breaker = 2                
            else:
                pos,dir = cii[0],cii[1]
                continue

    if breaker == 0:                                          # Depending on which process broke the loop, camera or leaf data recorded
        position_record = np.append(position_record,bi[1])
        direction_record = np.append(direction_record,bi[2])
        theta_1 = lambert(bi[2])
        x_1 = bi[1]
        dir = math.cos(theta_1),math.sin(theta_1)
        pos = x_1,0
        plop = intersect_camera(pos,dir)
        x_2 = abs(plop[0])
        if 0 <= x_2 <= lens_upper:
            x_3 = lens_transformation(theta_1,x_2)
        else: 
            x_3 = 10
        ccd_position = np.append(ccd_position, x_3)
    elif breaker == 1: 
        camera_pos = np.append(camera_pos,ci[1])
        camera_direction = np.append(camera_direction,ci[2])
        x__2 = ci[1]
        anglewangle = ci[2]
        x__3 = lens_transformation(anglewangle,x__2)
        ccd_position = np.append(ccd_position, x__3)
    else:
        continue
    counter += 1
    pbar.update(1)

pbar.close()
master_log = list(zip(logx,logy))
print(position_record)  # Prints the x position of absorption by the leaf
print(direction_record) # Prints the angle of absorption by the leaf
np.savetxt("position.csv", position_record, delimiter=",")      # Saves data to csvs
np.savetxt("direction.csv", direction_record, delimiter=",")
np.savetxt("camera_pos.csv", camera_pos, delimiter=",")
np.savetxt("camera_direction.csv", camera_direction, delimiter=",")
np.savetxt("ccdlog.csv",ccd_position, delimiter = ",")
#np.savetxt("log.csv", master_log, delimiter=",")

position_rec = np.genfromtxt('position.csv',delimiter=',')
im = Image.open('line.png', 'r')
pix_val = list(im.getdata())
intensity = []
pixel = []

# Cuts the image data down to exclude the edges
for i in range(len(pix_val)):
    tuple = pix_val[i]
    if 6745 <= i <= 6975 :
        intensity = np.append(intensity, tuple[0])
        pixel = np.append(pixel,i)
        
# Calculates fitted polynomial
avg = sum(intensity)/len(intensity)
maximum = max(intensity)
mm = pixel/9 - 762.2222222
relative_intensity = intensity/avg
p = np.polyfit(mm,relative_intensity, 2)
fit = p[0]*(mm**2) + p[1]*mm + p[2]


# Duplicates simulated data for axial symmetry
position_neg = []        
for i in range(len(position_rec)):
    item = (-1)*position_rec[i]
    position_neg = np.append(position_neg,item)
position_rec = np.append(position_rec,position_neg)

# Gets scatter data from histogram
bin_n = 100
n, bins, patches = plt.hist(position_rec, bins = bin_n)           # Gets histogram data
plt.show()
bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))] # Central x values of the bins
bins_mean2 =[]
for i in range(len(bins_mean)):
    if 8 <= i <= (bin_n-9):                                             # Removes sides so the data is within bounds of experimental
        value = bins_mean[i]
        bins_mean2 = np.append(bins_mean2, value*20)          # Converts to equal axes
mean_n = np.average(n)
normalised_n=[]
for i in range(len(n)):
    if 8 <= i <= (bin_n-9):
        value2 = n[i]
        normalised_n = np.append(normalised_n, (value2/mean_n))  # Normalisation
p2 = np.polyfit(bins_mean2,normalised_n, 4)
fit2 = p2[0]*(bins_mean2**4) + p2[1]*(bins_mean2**3) + p2[2]*(bins_mean2**2) + p2[3]*bins_mean2 + p2[4]

# Plot of simulated vs experimental data
#plt.scatter(mm,relative_intensity, s =5,label='Real Data')

plt.scatter(bins_mean2, normalised_n, s =5, color = 'r',label='Simulated Data')
ticks = (-15,-10,-5,0,15,10,5)
plt.plot(mm,fit, color = 'lime',linewidth = 3,label='Real Fit')
#plt.plot(bins_mean2,fit2, color = 'orange',linewidth = 3,label='Sim Fit')
plt.axvline(x=0, color = 'gray', ls = '--')
plt.xticks(ticks)
plt.ylim(0,2)
plt.title('Plot of Real Illumination Data')
plt.xlabel('Position from Centre [mm]')
plt.ylabel('Relative Intensity')
plt.legend()
plt.savefig('realdataplot.png',dpi = 1000)
plt.show()
np.savetxt("relative_intensity.csv", intensity, delimiter=",")
np.savetxt("mm.csv", mm, delimiter=",")
print('Real data fit: y =','{0:.3g}'.format(p[0]),'x^2  +','{0:.3g}'.format(p[1]),'x  +','{0:.3g}'.format(p[2]))
print('Simulation fit: y =','{0:.3g}'.format(p2[0]),'x^4  +','{0:.3g}'.format(p2[1]),'x^3  +','{0:.3g}'.format(p2[2]),'x^2  +','{0:.3g}'.format(p2[3]),'x  +','{0:.3g}'.format(p2[3]))
print('Where y is relative intensity, and x is position from centre')

# Values of merit
real_fit = p[0]*(bins_mean2**2) + p[1]*bins_mean2 + p[2]
chi_list = ((real_fit-normalised_n)**2)/real_fit
uniformity = ((normalised_n-1)**2)/1
chi_squared = sum(chi_list)
chi_uni = sum(uniformity)
r_cs_uni = chi_uni/99
r_cs = chi_squared/99
print('Experimental Reduced Chi-Squared =','{0:.4g}'.format(r_cs))
print('Uniformity Reduced Chi-Squared =','{0:.4g}'.format(r_cs_uni))
