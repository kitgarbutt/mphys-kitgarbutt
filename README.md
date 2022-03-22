# Enhancing Optical Imaging for Crop Science.
# 2D Simulation Code.
This is a 2D simulation of a 3D device, used to study bioflourescence in leaves. The initial geometry is shown below.

Photons emit from the LED, following a sinusoidal probability distribution, and reflect of various surfaces until they are absorbed by the leaf. The points on the x axis where the photons are absorbed are recorded and outputted. 

The purpose of this is to study the uniformity of illumination.
(In collaboration with @JackAYoung301)

<p align="left">
  <img src="274207123_5212102385487366_2221284332605063388_n.jpg" width="400" title="hover text">
</p>

Week 5 Changes:
 - Added partial specularity
 - Added absorption of camera
 - Added histograms of absorption position and angle data

Week 6 Changes:
 - Added 2D KDE plot
 - Changed baseline to Lambertian reflector
 - Added absorption factor into hemisphere and baseline
 - Fixed issue where camera_intersect wasn't working properly
 - Added position of interaction log
 - Started tracing rays graphically
