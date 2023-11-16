import pandas as pd
from xtgeo.surface import RegularSurface
import warmth
from warmth.data import haq87

#  |  get_dataframe(self, ijcolumns=False, ij=False, order='C', activeonly=True, fill_value=nan)
#  |      Return a Pandas dataframe object, with columns X_UTME, Y_UTMN, VALUES.
#  |  get_fence(self, xyfence: numpy.ndarray, sampling: Optional[str] = 'bilinear') -> numpy.ma.core.MaskedArray
#  |      Sample the surface along X and Y positions (numpy arrays) and get Z.

class SedimentStack:
    def __init__(self ):
        self.ms = []
        self.horizonDate = []

    def loadFromGridFiles(self, gridFileTimeTuples):
        self.ms = []
        self.horizonDate = []
        self.physicalProps = []
        for i,gft in enumerate(gridFileTimeTuples):
            mysurf = RegularSurface(gft[0])
            self.ms.append(mysurf)
            self.horizonDate.append(gft[1])
            if (i%3==0):
                self.physicalProps.append((1.500000, 2.301755e-09, 0.620000, 0.500, 2720.0, 2448.0))
            elif (i%3==1):
                self.physicalProps.append((1.538462, 2.079433e-09, 0.599730, 0.490, 2708.0, 2437.2))
            elif (i%3==2):
                self.physicalProps.append((1.904762, 4.719506e-10, 0.447705, 0.415, 2618.0, 2356.2))
        self.numLayers = len(self.ms)

    def getSurfaceByIndex(self, index):
        return self.ms[index]

    def verifyGrids(self):
        #
        # TODO: verify that the stack of grids have same geoextent, resolution, rotation, etc..
        #       verify that geotimes and depth are increasing, etc
        return True

    def createHorizonsAtIJ(self, i, j):
        # x,y = self.ms[0].get_xy_value_from_ij(i,j)
        horizons = []
        for msc in range(self.numLayers):
            base = self.ms[msc+1].get_xy_value_from_ij(i,j)[2] if (msc<self.numLayers -1) else self.ms[msc].get_xy_value_from_ij(i,j)[2]
            baseage = self.horizonDate[msc+1] if (msc<self.numLayers -1) else 0.0
            if (base is not None):
                base = base
            zz = self.ms[msc].get_xy_value_from_ij(i,j)[2]
            if (zz is not None):
                zz = zz
            hh = [ zz, self.horizonDate[msc], base, baseage]
            hh.extend(self.physicalProps[msc])
            horizons.append(hh)
        # print("dividing by 10!!")
        return horizons

    def create1DNodeAtIJ(self, i_ind, j_ind,):
        model = warmth.Model()
        node_template = model.builder.single_node_sediments_inputs_template.copy()
        horizons = self.createHorizonsAtIJ(i_ind, j_ind)
        for j,i in enumerate(horizons):
            row = pd.DataFrame({'top': i[0], 'topage': int(i[1]),
                                'k_cond': i[4], 'rhp': i[5], 'phi': i[6], 'decay': i[7], 'solidus': i[8], 'liquidus': i[9]}, index=[j])
            node_template = pd.concat( [node_template, row] )
        node = warmth.single_node()
        node.sediments_inputs = node_template
        #node.bflux = False
        #node.adiab = 0.0
        xpos, ypos = self.ms[0].get_xy_value_from_ij(i_ind, j_ind)[0:2]
        node.X = xpos
        node.Y = ypos
        return node

