import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

position_record = np.genfromtxt('position.csv',delimiter=',')
direction_record = np.genfromtxt('direction.csv',delimiter=',')
camera_pos = np.genfromtxt('camera_pos.csv',delimiter=',')
camera_direction = np.genfromtxt('camera_direction.csv',delimiter=',')


position_neg = []
                
for i in range(len(position_record)):
    item = (-1)*position_record[i]
    position_neg = np.append(position_neg,item)

position_record = np.append(position_record,position_neg)

direction_neg = []
                
for i in range(len(direction_record)):
    item = math.pi-direction_record[i]
    direction_neg = np.append(direction_neg,item)

direction_record = np.append(direction_record,direction_neg)

## POSITION ##

sns.displot(x = position_record, kde=True, bins =100, color = 'green')
plt.title('Leaf Absorption Position', fontsize =13)
plt.xlabel('Photon absorption position along x-axis',fontsize=12)
plt.ylabel('Number of photons', fontsize=12)
plt.savefig('position_histogram_leafsize.png', dpi=1000, bbox_inches='tight')
plt.show()

sns.displot(x = camera_pos, kde=True, bins =100, color = 'green')
plt.title('Camera Absorption Position', fontsize =13)
plt.xlabel('Photon absorption position along x-axis',fontsize=12)
plt.ylabel('Number of photons', fontsize=12)
plt.savefig('camera_pos_leafsize.png', dpi=1000, bbox_inches='tight')
plt.show()

## DIRECTION ##

sns.displot(x = direction_record, kde=True, bins =100, color = 'blue')
plt.title('Leaf Absorption Direction', fontsize =13)
plt.xlabel('Photon absorption angle [radians]',fontsize=12)
plt.ylabel('Number of photons', fontsize=12)
plt.savefig('direction_histogram_leafsize.png', dpi=1000, bbox_inches='tight')
plt.show()

sns.displot(x = camera_direction, kde=True, bins =100, color = 'blue')
plt.title('Camera Absorption Direction', fontsize =13)
plt.xlabel('Photon absorption angle [radians]',fontsize=12)
plt.ylabel('Number of photons', fontsize=12)
plt.savefig('camera_direction_leafsize.png', dpi=1000, bbox_inches='tight')
plt.show()
            
## 2D HISTOGRAM ##

plt.hist2d(position_record, direction_record, bins=(50, 50), cmap=plt.cm.jet)
plt.title('Absorption Properties of the Leaf', fontsize = 13)
plt.xlabel('Absorption position on x-axis [dm]',fontsize=12)
plt.ylabel('Angle of absorption [radians]', fontsize=12)
plt.savefig('2dhist_leafsize.png',dpi = 1000)
plt.colorbar()
plt.show()

sns.kdeplot(x = position_record, y = direction_record, fill=True, cmap = "magma", cbar = True, thresh =0)
plt.title('Absorption Properties of the Leaf', fontsize = 13)
plt.xlabel('Absorption position on x-axis [dm]',fontsize=12)
plt.ylabel('Angle of absorption [radians]', fontsize=12)
plt.xlim(-0.75,0.75)
plt.ylim(0,3.14)
plt.savefig('2dkde_leafsize.png',dpi = 1000)
plt.show()

plt.hist2d(camera_pos, camera_direction, bins=(50, 50), cmap=plt.cm.jet)
plt.title('Absorption Properties of camera', fontsize = 13)
plt.xlabel('Absorption position on x-axis [dm]',fontsize=12)
plt.ylabel('Angle of absorption [radians]', fontsize=12)
plt.savefig('2dhist_cam_leafsize.png',dpi = 1000)
plt.colorbar()
plt.show()

sns.kdeplot(x = camera_pos, y = camera_direction, fill=True, cmap = "viridis", cbar = True, thresh =0)
plt.title('Absorption Properties of the camera', fontsize = 13)
plt.xlabel('Absorption position on x-axis [dm]',fontsize=12)
plt.ylabel('Angle of absorption [radians]', fontsize=12)
plt.xlim(0,0.3)
plt.ylim(0,3.14)
plt.savefig('2dkde_cam_leafsize.png',dpi = 1000)
plt.show()

