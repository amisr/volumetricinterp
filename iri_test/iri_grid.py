# iri_grid.py

import numpy as np
import datetime as dt
from iri2016.base import IRI
import matplotlib.pyplot as plt

gdlat, gdlon = np.meshgrid(np.arange(-80., 80., 5.), np.arange(-180., 180., 5.))
altstart = 100.
altstop = 1000.
altstep = 5.
alt = np.arange(altstart, altstop+altstep, altstep)
time = dt.datetime(2017,2,24,18,0,0)

ne = np.empty((gdlat.shape[0], gdlat.shape[1], len(alt)))
for idx in np.ndindex(gdlat.shape):
    ne[idx] = IRI(time, (altstart, altstop, altstep), gdlat[idx], gdlon[idx]).ne

c = plt.pcolormesh(gdlon, gdlat, ne[:,:,40])
plt.colorbar(c)
plt.show()
