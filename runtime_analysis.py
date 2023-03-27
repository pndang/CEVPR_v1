import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

iterations = np.arange(1, 11, 1)

og_times = [3.8163402300000002, 
            3.5382675099999985,
            3.77384445,
            4.499156365999999,
            3.483695513999999,
            3.531312068000002,
            3.5300044899999987,
            3.549116459999991,
            3.678827141999991,
            3.8624800560000176]

og_plot = sns.lineplot(x=iterations, y=og_times)
plt.ylim((2, 5))
plt.xlabel('# iteration')
plt.ylabel('runtime, seconds')
plt.title('runtime analysis')
# plt.show()
