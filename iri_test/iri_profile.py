# iri_profile.py

import numpy as np
import datetime as dt
from iri2016.base import IRI

import matplotlib.pyplot as plt


glat = 74.7
glon = -94.9
alt = (100.,1000., 5.)
time = dt.datetime(2017,2,24,18,0,0)

iri = IRI(time, alt, glat, glon)
print(iri.ne)

plt.plot(iri.ne, np.arange(100.,1001., 5.))
plt.show()
