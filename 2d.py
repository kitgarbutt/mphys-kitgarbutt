import math
import random
import numpy as np
import matplotlib.pyplot as plt

# -------- Assembly Parameters -------- #

led_position = (0.9,0.05) # Position (x,y) of LED from origin
x_led = led_position[0]
y_led = led_position[1]
camera_position = 0.95 # Camera y-position
h_c = camera_position
R = 1
leaf_record = [12345]
leaf_width = 0.3

# -------- LED Emission Functions -------- #

def lambert():
    mn=0          # Lowest value of domain
    mx=math.pi    # Highest value of domain
    bound=1       # Upper bound of PDF value
    while True:
       x=random.uniform(mn,mx)      # Choose an x value inside the desired sampling domain.
       y=random.uniform(0,bound)    # Choose a x value between 0 and the maximum PDF value.
       pdf=math.sin(x)              # Calculate PDF
       if y<pdf:                    # Does (x,y) fall in the PDF?
           return x

def led():
  x = lambert()
  dir = math.cos(x), math.sin(x)     #Gets emission vector from lambertian distribution
  pos = led_position  #Gives initial position (Position of LED)
  return (pos,dir)

# -------- First Interaction -------- #
 
def first_interaction_surface(pos,dir):       
    surface_counter = []                            # Marks each interaction type (0 = vertical, 1 = camera, 2 = circle, 3 = base)
    a = R-y_led-h_c                                 # Distance in the y-direction from LED to camera
    theta_1 = math.atan(a/x_led)                    # Angle from -y direction to vertical-camera intersection, measured from LED.
    theta_2 = math.atan(a/(x_led-np.sqrt(1-a**2)))  # Angle from -y direction to camera-circle intersection, measured from LED.
    angle = lambert()
    if 0 < angle < (180 - theta_2): # If the angle meets these parameters, the photon emitted from LED will interact with the vertical first
      np.append(surface_counter, 0)                           # If this condition is met, add surface identifier 0 to surface_counter
    elif (180 - theta_2) <= angle <= (180 - theta_1): # If the angle meets these parameters, the photon emitted from LED will interact with the camera first
      np.append(surface_counter, 1)                                         # If this condition is met, add surface identifier 1 to surface_counter
    else:                             # If the angle meets these parameters, the photon emitted from LED will interact with the circle first
      np.append(surface_counter, 2)   # If this condition is met, add surface identifier 2 to surface_counter
    return(surface_counter)
        
def horizontal_specular(pos,dir):     # Gives new trajectory of photon that is specularly reflected on the horizontal surface
    new_dir = dir[0],-dir[1]
    return(pos,new_dir)
    
def on_or_off_leaf(pos,dir):          # Determines whether photon interacts with the leaf or the baseline
    if 0 < pos[0] < leaf_width:
        return 1
    if pos[0] > leaf_width:
        return 0

def reflect_or_absorb(pos,dir):       # Accounts for reflectivity of leaf
    if on_or_off_leaf(pos,dir) == 1:
        rand = random.random()
        if rand > 0.05:
            return 1
        else:
            return 0
    else:
        return 0
        
def base_interaction(pos,dir,leaf_record):            # Determines whether a photon hitting the x axis will be absorbed by the 
    if reflect_or_absorb(pos,dir) == 1:               # leaf or reflected, and also records x coordinate of absorption
        leaf_record = np.append(leaf_record, pos[0])
        return(leaf_record)
    else:
        return(horizontal_specular(pos,dir))
    
def circle_interaction(pos,dir):    # Function used to define the new position and direction of the photon after interaction with the 
    x = lambert()
    op = math.atan(pos[1]/pos[0])   # Angle between origin and point on circle
    alpha =  op + math.pi/2 + x     # New angle of trajectory
    dir = math.cos(alpha), math.sin(alpha) # Direction vector from angle
    return(pos,dir)
    
def vertical_interaction(pos,dir):  # Function used to define the new direction of the photon after interaction with the vertical
    dir = dir[0]*-1, dir[1]
    return(pos,dir)

def camera_interaction(pos,dir):    # Function used to define the new position and direction of the photon after interaction with the camera
    dir = dir[0], dir[1]*-1
    return(pos,dir)
    
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
  m = dir[1]/dir[0]                 # slope gradient
  c = pos[1] - m*pos[0]             # y intercept
  x = camera_position-c/m           # x coordinate of intersection with camera
  return(x,camera_position)

# -------- Subsequent Interactions -------- #
def interaction_surface(pos, dir):
    vertical = intersect_vertical(pos,dir)  #
    baseline = intersect_base(pos,dir)      # assigns each coordinate of interaction a variable.
    camera = intersect_camera(pos,dir)      #
    circle = intersect_circle(pos,dir)      #
    if len(circle) > 2:
      circle1 = circle[0],circle[1]      # 
      circle2 = circle[2],circle[3]      # in the case where the line crosses two positive points on the circle, 
    else:                                # this splits the variable into 2. otherwise it makes one automatically invalid.
      circle1 = circle                   #
      circle2 = -1,-1                     
    which = np.zeros([0,2])      # creates empty array
    if 0 < vertical[1] < 1:                                       #
      which = np.append(which, (0,vertical[1]))                   #
    if 0 < baseline[0] < 1:                                       # 
      which = np.append(which, (baseline[0],0))                   # This section tests whether each point lies within the 
    if 0 < camera[0] < np.sqrt(1-camera_position**2):             # boundaries of the geometry, and passes them based on this.
      which = np.append(which, (camera[0],camera_position))       #   
    if 0 < circle1[0] < 1 and 0 < circle1[1] < 1:                 # 
      which = np.append(which, (circle1[0],circle1[1]))           # 
    if 0 < circle2[0] < 1 and 0 < circle2[1] < 1:                 #
      which = np.append(which, (circle2[0],circle2[1]))           #
    pos1 = which[0],which[1]                    # Splits into two points (always going to be two points that are valid)
    pos2 = which[2],which[3]                    #
    delta1 = np.sqrt((pos[1]-pos1[1])**2 + (pos[0]-pos1[0])**2)   # For each point, determines the distance from initial position
    delta2 = np.sqrt((pos[1]-pos2[1])**2 + (pos[0]-pos2[0])**2)   #
    if delta1 > delta2:
      return(pos1)
    else:               # Point furthest from position selected 
      return(pos2)      # (when position is on a surface, one of the points will be that position)
      

#print(intersect_vertical(pos,dir))
#print(intersect_base(pos,dir))
#print(intersect_camera(pos,dir))
#print(intersect_circle(pos,dir))
#print(interaction_surface(pos, dir))

## -------- SCRIPT -------- ##
counter = 0
record =[]

while counter < 1000:

  pos,dir = led() # Defining the emission position and direction of the photon
  surface_counter = first_interaction_surface(pos,dir) # Determining the first interaction surface of the photon, recording it as a number

  if surface_counter == 0: # If first interaction is with vertical, call relevant intersection and interaction functions
      pos1 = intersect_vertical(pos,dir)
      pos,dir = vertical_interaction(pos1,dir)
  elif surface_counter == 1: # If first interaction is with camera module, call relevant intersection and interaction functions
      pos1 = intersect_camera(pos,dir)
      pos,dir = camera_interaction(pos1,dir)
  elif surface_counter == 2: # If first interaction is with circle, call relevant intersection and interaction functions
      pos1 = intersect_circle(pos,dir)
      pos,dir = circle_interaction(pos1,dir)

  absorbed = 0
  while absorbed == 0:
    pos = interaction_surface(pos,dir)              # Redefine pos as new position
    if pos[0] == 0:
      pos, dir = vertical_interaction(pos,dir)
      continue                                      # If x = 0, vertical function used for new direction, redefines dir
    if pos[1] == 0:
      bi = base_interaction(pos,dir,leaf_record)    # If y = 0, horizontal function used for new direction, redefines dir
      if bi[0] == 12345:                            # If output is leaf_record, photon absorbed and loop stopped
        absorbed =+ 1
      else:
        pos,dir = bi 
        continue                                    # Or new direction, redefines dir
    if pos[1] == camera_position:
      pos,dir = camera_interaction(pos,dir)
      continue
    else:
      pos,dir = circle_interaction(pos,dir)
      continue
  record = np.append(record,bi[1])
  counter += 1

print(record)      # Prints the x position of absorbtion by the leaf
