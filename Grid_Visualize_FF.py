# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:42:58 2022

@author: ffaraj
"""

#%% Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% Load Data
Data = pd.read_csv('Arsenic_Data.csv')
East = Data['East']
North = Data['North']
As = Data['Arsenic']


#%% Create simple visual
plt.rcParams["font.family"] = "arial"
S=22
lw = 1
plt.rcParams["font.size"] = S
plt.close()
fig, ax = plt.subplots()
fign='Spatial_Data'

plt.scatter(East,North,c=As,cmap='jet',edgecolor='k',linewidth=0.75,s=50)

plt.clim(2,22)

plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')


cbar =plt.colorbar(aspect=15)
cbar.set_label('Arsenic concentration [µg/L]')
cbar.outline.set_linewidth(lw)
cbar.ax.tick_params(which='both',direction='in',width=lw) 



ax.set_axisbelow(True)
ax.tick_params(axis="both",direction="in",bottom=True, top=True, left=True, right=True,width=lw)
ax.tick_params(which='minor',axis="both",direction="in",bottom=True, top=True, left=True, right=True,width=lw)




ax.spines['right'].set_linewidth(lw)
ax.spines['top'].set_linewidth(lw)
ax.spines['bottom'].set_linewidth(lw)
ax.spines['left'].set_linewidth(lw)

plt.gca().set_aspect('equal', adjustable='box')

plt.xlim(0,1000)
plt.ylim(0,1000)


plt.savefig(fign, bbox_inches='tight', dpi=300,transparent=True)


#%% Generate grid

x = np.linspace(50,950,10)
y = np.linspace(50,950,10)

[X,Y] = np.meshgrid(x,y)

# Z = np.gridata(x,y,As,X,Y)
Xg = np.concatenate(X)
Yg = np.concatenate(Y)


#%% Visualize grid
plt.rcParams["font.family"] = "arial"
S=16
lw = 1
plt.rcParams["font.size"] = S
plt.close()
fig, ax = plt.subplots()
fign='Spatial_Grid_Data'

plt.scatter(Xg,Yg,marker='s',s=715,edgecolor='k',linewidth=0.5)

plt.scatter(East,North,c=As,cmap='jet',edgecolor='k',linewidth=0.75,s=50)

plt.clim(2,22)



plt.clim(2,22)

plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')


cbar =plt.colorbar(aspect=15)
cbar.set_label('Arsenic concentration [µg/L]')
cbar.outline.set_linewidth(lw)
cbar.ax.tick_params(which='both',direction='in',width=lw) 



ax.set_axisbelow(False)
ax.tick_params(axis="both",direction="in",bottom=True, top=True, left=True, right=True,width=lw)
ax.tick_params(which='minor',axis="both",direction="in",bottom=True, top=True, left=True, right=True,width=lw)




ax.spines['right'].set_linewidth(lw)
ax.spines['top'].set_linewidth(lw)
ax.spines['bottom'].set_linewidth(lw)
ax.spines['left'].set_linewidth(lw)

plt.gca().set_aspect('equal', adjustable='box')

plt.xlim(0,1000)
plt.ylim(0,1000)


plt.savefig(fign, bbox_inches='tight', dpi=300,transparent=True)




#%% IDW to grid


SEARCH_RADIUS = 80.0 #m
from scipy.spatial import cKDTree
from statistics import mode

P = np.array([East,North]).T
PQ = np.array([Xg,Yg]).T
kdt = cKDTree(P)
Idx_all = pd.DataFrame.from_records(kdt.query_ball_point(PQ,SEARCH_RADIUS))
dst, Idx = kdt.query(PQ, k=np.size(np.max(Idx_all)), distance_upper_bound=SEARCH_RADIUS)

Idx = pd.DataFrame( np.array(Idx,dtype=object))
dst = pd.DataFrame( np.array(dst,dtype=object))
Idx = Idx.replace(np.max(Idx),np.NaN)
dst = dst.replace(np.inf,np.NaN)

IDW_As = pd.Series(len(Xg))
Count = pd.Series(len(Xg))
for i in range(0,len(Xg)):
    DIST = dst.iloc[i].dropna()
    DIST.name = 'Dist'
    INDX = Idx.iloc[i].dropna()
    INDX.name = 'Indx'
    DISTINDX = pd.concat([DIST, INDX], axis=1).dropna()
    DIST = DISTINDX['Dist']
    INDX = DISTINDX['Indx']
    
    IDW_As[i]= (np.sum(np.array(As[INDX]) /DIST) / np.sum(1/DIST))
    Count[i]=len(INDX)

# IDW_As = IDW_As.fillna(0)

#%% Visualize grid
BM_ms = 708
plt.rcParams["font.family"] = "arial"
S=18
lw = 1
plt.rcParams["font.size"] = S
plt.close()
fig, ax = plt.subplots()
fign='Spatial_Grid_Data'
# plt.scatter(Xg[IDW_As.isna()==True],Yg[IDW_As.isna()==True],marker='s',c='w',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='jet')

plt.scatter(Xg,Yg,c=IDW_As,marker='s',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='jet')
plt.clim(2,22)
# plt.scatter(Xg,Yg,marker='s',s=715,edgecolor='k',linewidth=0.5)

plt.scatter(East,North,c=As,cmap='jet',edgecolor='k',linewidth=0.75,s=50)

plt.clim(2,22)



plt.clim(2,22)

plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')
plt.xticks([0,200,400,600,800,1000])

cbar =plt.colorbar(aspect=15)
cbar.set_label('Arsenic concentration [µg/L]')
cbar.outline.set_linewidth(lw)
cbar.ax.tick_params(which='both',direction='in',width=lw) 



ax.set_axisbelow(False)
ax.tick_params(axis="both",direction="in",bottom=True, top=True, left=True, right=True,width=lw)
ax.tick_params(which='minor',axis="both",direction="in",bottom=True, top=True, left=True, right=True,width=lw)




ax.spines['right'].set_linewidth(lw)
ax.spines['top'].set_linewidth(lw)
ax.spines['bottom'].set_linewidth(lw)
ax.spines['left'].set_linewidth(lw)

plt.gca().set_aspect('equal', adjustable='box')

plt.xlim(0,1000)
plt.ylim(0,1000)


plt.savefig(fign, bbox_inches='tight', dpi=300,transparent=True)





#%% Visualize grid above limit below limit
BM_ms = 708
plt.rcParams["font.family"] = "arial"
S=18
lw = 1
plt.rcParams["font.size"] = S
plt.close()
fig, ax = plt.subplots()
fign='Spatial_Grid_Data'
# plt.scatter(Xg[IDW_As.isna()==True],Yg[IDW_As.isna()==True],marker='s',c='w',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='jet')

plt.scatter(Xg[IDW_As>10],Yg[IDW_As>10],c='r',marker='s',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='jet')
plt.scatter(Xg[IDW_As<=10],Yg[IDW_As<=10],c='g',marker='s',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='jet')
# plt.scatter(Xg,Yg,marker='s',s=715,edgecolor='k',linewidth=0.5)

# plt.scatter(East,North,c='k',marker='x',cmap='jet',edgecolor='k',linewidth=0.75,s=50)

plt.scatter(East[As>10],North[As>10],c='r',s=50,edgecolor='k',linewidth=0.75,cmap='jet')
plt.scatter(East[As<=10],North[As<=10],c='g',s=50,edgecolor='k',linewidth=0.75,cmap='jet')


# plt.clim(2,22)



plt.clim(2,22)

plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')


# cbar =plt.colorbar(aspect=15)
# cbar.set_label('Arsenic concentration [µg/L]')
# cbar.outline.set_linewidth(lw)
# cbar.ax.tick_params(which='both',direction='in',width=lw) 


plt.xticks([0,200,400,600,800,1000])

ax.set_axisbelow(False)
ax.tick_params(axis="both",direction="in",bottom=True, top=True, left=True, right=True,width=lw)
ax.tick_params(which='minor',axis="both",direction="in",bottom=True, top=True, left=True, right=True,width=lw)




ax.spines['right'].set_linewidth(lw)
ax.spines['top'].set_linewidth(lw)
ax.spines['bottom'].set_linewidth(lw)
ax.spines['left'].set_linewidth(lw)

plt.gca().set_aspect('equal', adjustable='box')

plt.xlim(0,1000)
plt.ylim(0,1000)


plt.savefig(fign, bbox_inches='tight', dpi=300,transparent=True)





#%% Visualize grid warning
BM_ms = 708
plt.rcParams["font.family"] = "arial"
S=18
lw = 1
plt.rcParams["font.size"] = S
plt.close()
fig, ax = plt.subplots()
fign='Spatial_Grid_Data'
# plt.scatter(Xg[IDW_As.isna()==True],Yg[IDW_As.isna()==True],marker='s',c='w',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='jet')

plt.scatter(Xg[IDW_As>10],Yg[IDW_As>10],c='r',marker='s',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='jet')
plt.scatter(Xg[IDW_As<=8],Yg[IDW_As<=8],c='g',marker='s',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='jet')
plt.scatter(Xg[(IDW_As>8) & (IDW_As<10)],Yg[(IDW_As>8) & (IDW_As<10)],c='y',marker='s',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='jet')
# plt.scatter(Xg,Yg,marker='s',s=715,edgecolor='k',linewidth=0.5)

plt.scatter(East,North,c='k',marker='x',cmap='jet',edgecolor='k',linewidth=0.75,s=50)

# plt.clim(2,22)



plt.clim(2,22)

plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')


# cbar =plt.colorbar(aspect=15)
# cbar.set_label('Arsenic concentration [µg/L]')
# cbar.outline.set_linewidth(lw)
# cbar.ax.tick_params(which='both',direction='in',width=lw) 

plt.xticks([0,200,400,600,800,1000])


ax.set_axisbelow(False)
ax.tick_params(axis="both",direction="in",bottom=True, top=True, left=True, right=True,width=lw)
ax.tick_params(which='minor',axis="both",direction="in",bottom=True, top=True, left=True, right=True,width=lw)




ax.spines['right'].set_linewidth(lw)
ax.spines['top'].set_linewidth(lw)
ax.spines['bottom'].set_linewidth(lw)
ax.spines['left'].set_linewidth(lw)

plt.gca().set_aspect('equal', adjustable='box')

plt.xlim(0,1000)
plt.ylim(0,1000)


plt.savefig(fign, bbox_inches='tight', dpi=300,transparent=True)






#%% Visualize grid count
BM_ms = 708
plt.rcParams["font.family"] = "arial"
S=18
lw = 1
plt.rcParams["font.size"] = S
plt.close()
fig, ax = plt.subplots()
fign='Spatial_Grid_Data'
# plt.scatter(Xg[IDW_As.isna()==True],Yg[IDW_As.isna()==True],marker='s',c='w',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='jet')

plt.scatter(Xg[(Count<1) ],Yg[(Count<1)],c='w',marker='s',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='viridis')
plt.scatter(Xg[(Count>0) & (Count<=2)],Yg[(Count>0) & (Count<=2)],c='#472F7D',marker='s',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='viridis')
plt.scatter(Xg[(Count>2) & (Count<=6)],Yg[(Count>2) & (Count<=6)],c='#26AD81',marker='s',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='viridis')
plt.scatter(Xg[(Count>6) ],Yg[(Count>6)],c='#F6E620',marker='s',s=BM_ms,edgecolor='k',linewidth=0.5,cmap='viridis')

plt.clim(1,8)

# cbar =plt.colorbar(aspect=15)
# cbar.set_label('Samples per block centroid')
# cbar.outline.set_linewidth(lw)
# cbar.ax.tick_params(which='both',direction='in',width=lw) 

plt.scatter(East,North,c='k',marker='x',cmap='jet',edgecolor='k',linewidth=1.75,s=50)

# plt.clim(2,22)
# plt.scatter(Xg,Yg,marker='s',s=715,edgecolor='k',linewidth=0.5)

# plt.scatter(East,North,c=As,cmap='jet',edgecolor='k',linewidth=0.75,s=50)

# plt.clim(2,22)



# plt.clim(2,22)

plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')




plt.xticks([0,200,400,600,800,1000])

ax.set_axisbelow(False)
ax.tick_params(axis="both",direction="in",bottom=True, top=True, left=True, right=True,width=lw)
ax.tick_params(which='minor',axis="both",direction="in",bottom=True, top=True, left=True, right=True,width=lw)




ax.spines['right'].set_linewidth(lw)
ax.spines['top'].set_linewidth(lw)
ax.spines['bottom'].set_linewidth(lw)
ax.spines['left'].set_linewidth(lw)

plt.gca().set_aspect('equal', adjustable='box')

plt.xlim(0,1000)
plt.ylim(0,1000)


plt.savefig(fign, bbox_inches='tight', dpi=300,transparent=True)




#%% pie
plt.rcParams["font.family"] = "arial"
S=18
lw = 1
plt.rcParams["font.size"] = S
plt.close()
fig, ax = plt.subplots()
fign='pie'



p1 = plt.pie([59,41],wedgeprops={"edgecolor":"k",'linewidth': 0.75},colors=['r','g'])

plt.savefig(fign, bbox_inches='tight', dpi=300,transparent=True)
