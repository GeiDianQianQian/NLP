import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

algo=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dev=[0.510169, 0.512868, 0.517365, 0.519008, 0.520103, 0.523115, 0.526166, 0.530742, 0.537038, 0.549867]
test=[0.529, 0.529, 0.533, 0.539, 0.541, 0.546, 0.547, 0.548, 0.555, 0.564]
time=[93, 11, 50, 48, 55, 291, 343, 377, 377, 870]

plt.bar(algo, time, label='Run Time')
#plt.axis([0, 10, 0, 10])
#plt.plot(algo, dev, label='Dev Score')
#plt.plot(algo, test, label='Test Score')
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.ylabel('Running Time (Seconds)')
plt.legend()
plt.xlabel('Algorithm id')
plt.show()