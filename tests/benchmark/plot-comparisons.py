import xtgeo
import numpy as np

# %matplotlib
import matplotlib as plt
plt.use('TkAgg')

def getArrayN(T,p, subind):
    def floatKey2D(xp):
        # xpI = (xp[0]>=0.0) 
        return( int((xp[0]+0.01)*10), int((xp[1]+0.01)*10) )
    values = {}
    for i,pp in enumerate(p):
        fkey = floatKey2D(pp)
        nn = values.get(fkey,[])
        nn.append( (pp[2],T[i]) )
        values[fkey] = nn
    Ts = np.ones([ms[0].nx,ms[0].ny]) * np.nan
    Zs = np.ones([ms[0].nx,ms[0].ny]) * np.nan
    for k,v in values.items():
        zz = [a[0] for a in v]
        vv = [a[1] for a in v]
        ind = np.array(zz).argsort()
        xi = int(0.5+k[0]/1000)
        yi = int(0.5+k[1]/1000)
        if (xi>=0) and (xi<Ts.shape[0]) and (yi>=0) and (yi<Ts.shape[1]):
            Ts[xi,yi] = vv[ind[subind]]
            Zs[xi,yi] = zz[ind[subind]]
    return Ts, Zs


#
# Temperature
#
ms = []
ms.append(xtgeo.surface_from_file('results-hypothetical\\layercake_all_sand\\Top of model - Temperature - 0 Ma.gri'))
ms.append(xtgeo.surface_from_file('results-hypothetical\\layercake_all_sand\\30Ma - Temperature - 0 Ma.gri'))
# ms.append(xtgeo.surface_from_file('results-hypothetical\\layercake_all_sand\\Top of 50MA - Temperature - 0 Ma.gri'))
ms.append(xtgeo.surface_from_file('results-hypothetical\\layercake_all_sand\\Base of model - Temperature - 0 Ma.gri'))

T0 = np.load('out-hypothetical/T-value-0.npy')
p0 = np.load('out-hypothetical/mesh-pos-0.npy')
Ts0, Zs0 = getArrayN(T0,p0,0)
Ts1, Zs1 = getArrayN(T0,p0,1)
Ts2, Zs2 = getArrayN(T0,p0,2)

fig, ax = plt.pyplot.subplots(2, 2)
(ax1,ax2,ax3,ax4) = np.reshape(ax,[4])

fig.suptitle('Layercake all sand -- Temperature differences')
im1 = ax1.imshow( Ts0-ms[0].values)
ax1.set_title('Difference Petromod vs. SH3D, Top')
cbar1 = fig.colorbar(im1, ax=ax1)

im2 = ax2.imshow( Ts1-ms[1].values)
ax2.set_title('Difference Petromod vs. SH3D, 30 Ma')
cbar2 = fig.colorbar(im2, ax=ax2)

im3 = ax3.imshow( Ts2-ms[2].values)
ax3.set_title('Difference Petromod vs. SH3D, Base')
cbar3 = fig.colorbar(im3, ax=ax3)

zz = np.array([np.mean(Zs0), np.mean(Zs1), np.mean(Zs2)]) * (-1) - 2000
plt.pyplot.plot( [np.mean(Ts0), np.mean(Ts1), np.mean(Ts2)], zz, label="SH3D")
plt.pyplot.plot( [np.mean(ms[0].values), np.mean(ms[1].values), np.mean(ms[2].values)],zz, label="PM")
plt.pyplot.legend(loc="lower left", prop={'size': 7})
plt.pyplot.show()



#
# Conductivity
#
ms = []
ms.append(xtgeo.surface_from_file('results-hypothetical\\layercake_all_sand_isotropic\\Top of model- Vertical Thermal Conductivity - 0 Ma.gri'))
ms.append(xtgeo.surface_from_file('results-hypothetical\\layercake_all_sand_isotropic\\Top of 30 - Vertical Thermal Conductivity - 0 Ma.gri'))
# ms.append(xtgeo.surface_from_file('results-hypothetical\\layercake_all_sand_isotropic\\Top of 50MA - Temperature - 0 Ma.gri'))
ms.append(xtgeo.surface_from_file('results-hypothetical\\layercake_all_sand_isotropic\\Base of mode - Vertical Thermal Conductivity - 0 Ma.gri'))


C0 = np.load('out-hypothetical/k-value-0.npy')
pp = np.load('out-hypothetical/cell-pos-0.npy')

Cs0, Zs0 = getArrayN(C0,pp,0)
Cs1, Zs1 = getArrayN(C0,pp,1)

fig, ax = plt.pyplot.subplots(3, 2)
(ax1,ax2,ax3,ax4,ax5,ax6) = np.reshape(ax,[6])

fig.suptitle('Layercake all sand -- Conductivity')
im1 = ax1.imshow( Cs0 )
ax1.set_title('SH3D 0 (top-30MA)')
cbar1 = fig.colorbar(im1, ax=ax1)

im2 = ax2.imshow( Cs1 )
ax2.set_title('SH3D 1 (30MA-Base)')
cbar2 = fig.colorbar(im2, ax=ax2)

im3 = ax3.imshow( ms[0].values )
ax3.set_title('Petromod Top')
cbar3 = fig.colorbar(im3, ax=ax3)

im4 = ax4.imshow( ms[1].values )
ax4.set_title('Petromod 30MA')
cbar4 = fig.colorbar(im4, ax=ax4)

im5 = ax5.imshow( ms[2].values )
ax5.set_title('Petromod Base')
cbar5 = fig.colorbar(im5, ax=ax5)
plt.pyplot.show()


