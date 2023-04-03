import sys
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Runtime profiling code
#         n = 50
#         times = []
#         for ite in range(n):
#           task_name = f"task_{ite+1}"
#           start_1 = timer()
#           # Calculating rolling averages for each variant dataframes in dictionary 
#           for key, value in variants_dict.items():
#               value.reset_index(inplace=True)
#               calculate_rolling_avg(value, 5, smoothing_period)
#               value.set_index('index', inplace=True)
#           # dash.callback_context.record_timing(task_name, timer() - start_1, '1st task')
#           time = timer() - start_1
#           times.append(time)
#           print(f"Iteration {ite+1}/{n}: {time}")                
#         print(np.mean(times))

iterations = np.arange(1, 11, 1)

og_times = np.array([3.8163402300000002, 
                     3.5382675099999985,
                     3.77384445,
                     3.752621731999999,
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
plt.ylabel('avg runtime, seconds')
plt.suptitle('App Runtime Analysis', fontsize=18)
plt.title('Change: new smoothing algorithm using queue data structure', fontsize=10)
plt.legend(['initial ~ row iteration', 'optimized ~ queue'])

plt.savefig("runtime_analysis.png")

initial = np.mean(og_times)
new = np.mean(new_times)

diff = abs(initial - new)
diff_percentage = (diff / initial)*100

print(f"Initial: {initial}, New: {new}")
print(diff)
print(diff_percentage)

# Initial avg: 3.6516509651999995, New avg: 0.12209832960000233
# Difference: 3.5295526355999973
# Percentage difference: 96.65635268092181
