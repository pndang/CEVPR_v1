import sys
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

iterations = np.arange(1, 11, 1)

og_times = np.array([3.8163402300000002, 
                     3.5382675099999985,
                     3.77384445,
                     4.499156365999999,
                     3.483695513999999,
                     3.531312068000002,
                     3.5300044899999987,
                     3.549116459999991,
                     3.678827141999991,
                     3.8624800560000176])

new_times = np.array([0.13816570400002093,
                      0.12149308599999997,
                      0.12084483399999925,
                      0.11002622600000024,
                      0.11927368600000193,
                      0.10989246199999911,
                      0.1468550860000073,
                      0.10848503400000027,
                      0.11298818200000596,
                      0.13295899599998848])

plt.plot(iterations, og_times, iterations, new_times)
plt.ylim((0, 5.5))
plt.xlabel('# iteration')
plt.ylabel('runtime, seconds')
plt.title('runtime analysis')

plt.savefig("runtime_analysis.png")
