#Program to generate a grid defining a droplet of given dimensions
from math import *


print( 'Generating Grid.......')

ghost_start_index=0
ghost_end_index=0

part_spacing=0.01/25
base_len=0.02;
height=0.01
#GRID GENERATOR CODE FROM HERE
file = open('grid.txt','w')
timestep=1;
file.write(str(timestep)+'\n')
length=int(base_len/part_spacing)*part_spacing
file.write(str(length)+'\n')
hsml=1.2*part_spacing
file.write(str(part_spacing)+'\n')
ntotal=0
string=''
totmass=0

nlength=int(length/part_spacing)
nheight=int(1.1*height/part_spacing)


#Variables used to scale dimesions for rendering.
RenderXScale=length
RenderYScale=(height+3*part_spacing)
RenderXOffset=0
RenderYOffset=0
if RenderXScale>RenderYScale:
	RenderXOffset=-0.5
	RenderYOffset=-0.5*RenderYScale/RenderXScale
	RenderYScale=RenderXScale
else:
	RenderXOffset=-RenderXScale/RenderYScale*0.5
	RenderYOffset=-0.5
	RenderXScale=RenderYScale

#Setting 3 layers of wall particles underneath the droplet
for i in range(2*nlength):
	x=part_spacing*(i)*0.5
	vx=0
	vy=-0
	rho=1000
	eta=0.001
	pressure = 0
	mass = rho*part_spacing**2
	y=-part_spacing
	string=string + str(x) + ' ' + str(y) + ' ' + str(vx) + ' ' + str(vy) + ' ' + str(rho) + ' ' + str(rho) + ' ' + str(pressure) + ' ' + str(mass) + ' ' + str(eta) + ' ' + str(hsml) +  ' ' + str(1)+ ' ' + '0'+ ' ' + '1'+ '\n'
	y=-2*part_spacing
	string=string + str(x) + ' ' + str(y) + ' ' + str(vx) + ' ' + str(vy) + ' ' + str(rho) + ' ' + str(rho) + ' ' + str(pressure) + ' ' + str(mass) + ' ' + str(eta) + ' ' + str(hsml) +  ' ' + str(1)+ ' ' + '0'+ ' ' + '1'+ '\n'
	y=-3*part_spacing
	string=string + str(x) + ' ' + str(y) + ' ' + str(vx) + ' ' + str(vy) + ' ' + str(rho) + ' ' + str(rho) + ' ' + str(pressure) + ' ' + str(mass) + ' ' + str(eta) + ' ' + str(hsml) +  ' ' + str(1)+ ' ' + '0'+ ' ' + '1'+ '\n'
	ntotal=ntotal+3
	totmass+=3*mass

for i in range(2*nheight):
	y=part_spacing*i*0.5
	vx=0
	vy=-0
	rho=1000
	eta=0.001
	pressure = 0
	mass = rho*part_spacing**2
	x=0;
	string=string + str(x) + ' ' + str(y) + ' ' + str(vx) + ' ' + str(vy) + ' ' + str(rho) + ' ' + str(rho) + ' ' + str(pressure) + ' ' + str(mass) + ' ' + str(eta) + ' ' + str(hsml) +  ' ' + str(1)+ ' ' + '0'+ ' ' + '1'+ '\n'
	x=part_spacing
	string=string + str(x) + ' ' + str(y) + ' ' + str(vx) + ' ' + str(vy) + ' ' + str(rho) + ' ' + str(rho) + ' ' + str(pressure) + ' ' + str(mass) + ' ' + str(eta) + ' ' + str(hsml) +  ' ' + str(1)+ ' ' + '0'+ ' ' + '1'+ '\n'
	x=2*part_spacing
	string=string + str(x) + ' ' + str(y) + ' ' + str(vx) + ' ' + str(vy) + ' ' + str(rho) + ' ' + str(rho) + ' ' + str(pressure) + ' ' + str(mass) + ' ' + str(eta) + ' ' + str(hsml) +  ' ' + str(1)+ ' ' + '0'+ ' ' + '1'+ '\n'
	x=part_spacing*(nlength-1)
	string=string + str(x) + ' ' + str(y) + ' ' + str(vx) + ' ' + str(vy) + ' ' + str(rho) + ' ' + str(rho) + ' ' + str(pressure) + ' ' + str(mass) + ' ' + str(eta) + ' ' + str(hsml) +  ' ' + str(1)+ ' ' + '0'+ ' ' + '1'+ '\n'
	x=part_spacing*(nlength-2)
	string=string + str(x) + ' ' + str(y) + ' ' + str(vx) + ' ' + str(vy) + ' ' + str(rho) + ' ' + str(rho) + ' ' + str(pressure) + ' ' + str(mass) + ' ' + str(eta) + ' ' + str(hsml) +  ' ' + str(1)+ ' ' + '0'+ ' ' + '1'+ '\n'
	x=part_spacing*(nlength-3)
	string=string + str(x) + ' ' + str(y) + ' ' + str(vx) + ' ' + str(vy) + ' ' + str(rho) + ' ' + str(rho) + ' ' + str(pressure) + ' ' + str(mass) + ' ' + str(eta) + ' ' + str(hsml) +  ' ' + str(1)+ ' ' + '0'+ ' ' + '1'+ '\n'
	ntotal+=6;
	totmass+=6*mass
print('yp')
#Fluid particles
for i in range(nheight):
	for j in range(3,nheight):
		x=part_spacing*(j)
		y=part_spacing*(i)
		vx=0
		vy=-0
		pressure = 0
		hsml=1.2*part_spacing
		rho=1000
		eta=0.001
		fluidcode=1
		mass=rho*part_spacing**2
		string=string + str(x) + ' ' + str(y) + ' ' + str(vx) + ' ' + str(vy) + ' ' + str(rho) + ' ' + str(rho) + ' ' + str(pressure) + ' ' + str(mass) + ' ' + str(eta) + ' ' + str(hsml) +  ' ' + str(fluidcode)+ ' ' + '0'+ ' ' + '0'+ '\n'
		ntotal=ntotal+1
		totmass+=mass
file.write(str(ntotal)+'\n'+string+'0\n'+'0\n'+'0\n')

contact_angle=0
radius=0
volume=0


#details written to file
file.write('####################################################\n')
file.write('Contact Angle: '+str(contact_angle))
file.write('\nRadius: '+str(radius))
file.write('\nVolume: '+str(volume))
file.write('\nHeight: '+str(height))
file.write('\nGhost start index: '+str(ghost_start_index))
file.write('\nGhost end index: '+str(ghost_end_index))
file.write('\n####################################################')

file.close()
file=open('config.h','w')
file.write('#define GHOST_END_INDEX '+str(ghost_end_index))
file.write('\n#define GHOST_START_INDEX '+str(ghost_start_index))
file.write('\n#define RENDER_X_SCALE '+str(RenderXScale))
file.write('\n#define RENDER_Y_SCALE '+str(RenderYScale))
file.write('\n#define RENDER_X_OFFSET '+str(RenderXOffset))
file.write('\n#define RENDER_Y_OFFSET '+str(RenderYOffset))
file.close()
print( 'Done.\n')
print( 'Ghost start index:',ghost_start_index)
print( 'Ghost end index:',ghost_end_index,'\n')

print( 'Other details at the end of the file.\n')
tmp=input()

