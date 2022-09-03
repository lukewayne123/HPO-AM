from pylab import *
import brewer2mpl
import numpy as np
from matplotlib import pyplot as plt
import csv

import matplotlib.ticker as ticker

# Folder = 'NNslt40flippedduration0'
root_folder ='sptfullsa040advFlip'
# gamename = 'cartpole'
# gamename = 'mountaincar'
# gamename = 'acrobot'
gamename = 'lunarlander'
folder_list = [gamename+'vanilla',gamename+'spt080']
# folder_list = ['NNslt40flippedduration0','NNrc40flippedduration0','NN0flippedduration0']
# folder_list = ['NNsadependent_slt','NNsadependent_rc','NNsadependent_vanilla']
# seed_list = ['289','666','517','789']
seed_list = ['1','2','3','4','5']
DataLength = [1] * len(seed_list)
def load_csv(file_path, x_scale = 1):
    with open(file_path,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        x = []
        y = []
        is_skipped_scheme = False
        for row in plots:
            if not is_skipped_scheme:
                is_skipped_scheme = True
                continue
            # x.append(int(row[1])//x_scale)
            x.append(int(row[1]) )
            y.append(float(row[2]))
            # if Target == "Lambda" and y[-1] < 0:
            #     y[-1] = 0
        # average_x = []
        # average_y = []
        # for i in range(len(x)):
        #     x_sum = 0.0
        #     y_sum = 0.0
        #     avg = 50 # DMLab
        #     if Folder == "Data":
        #         avg = 2 # ML agent

        #     if i % avg == 0 and i+avg < len(x) and y[i+1] < 2e8:
        #         for idx in range(avg):
        #             x_sum += x[i + idx]
        #             y_sum += y[i + idx]
        #         average_x.append(x_sum / avg)
        #         average_y.append(y_sum / avg)
        #     #if i % 2 == 0 and  i+1 < len(x) and y[i+1] < 10000000: #  for ML-Agents
        #     #    average_x.append((x[i] + x[i+1])/2)
        #     #    average_y.append((y[i] + y[i+1])/2)
        #     #if i % 2 == 0 and  i+1 < len(x) and y[i+1] < 2e8:
        #     #    average_x.append((x[i] + x[i+1])/2)
        #     #    average_y.append((y[i] + y[i+1])/2)
        #     # average_x.append(x[i])
        #     # average_y.append(y[i])
        # return (x, y, average_x, average_y)
        return (x, y)
def refine_data(datas):
    refined_x = []
    refined_y = []
    min_y = []
    max_y = []

    max_len = 0
    for data in datas:
        if len(data[0]) > max_len:
            max_len = len(data[0])
            refined_x = data[0]
    for i in range(max_len):
        ys = []
        for data in datas:
            if i < len(data[1]):
                ys.append(data[1][i])
        refined_y.append(np.mean(ys))
        min_y.append(np.min(ys))
        max_y.append(np.max(ys))
    return (refined_x, refined_y, min_y, max_y)

bmap = brewer2mpl.get_map('Set1', 'qualitative', 8) # RBGPOY
bmap2 = brewer2mpl.get_map('Set2', 'qualitative', 8)
colors = bmap.mpl_colors

legend_colors = bmap.mpl_colors

human_color = bmap2.mpl_colors[1]

fig = figure(figsize=(14,9))  # no frame
ax = fig.add_subplot(111)


data_sets = []
# for x in range(DataLength[idx]):
for idx in range( len(folder_list) ):
    Folder =  folder_list[idx]
    samples = []
    for x in range( len(seed_list) ):
        samples.append(load_csv("{0}/{1}/{2}.csv".format(root_folder ,Folder, seed_list[x] ), x_scale=1))
    print( len(samples) )
    print( len(samples[0]) )
    print( len(samples[0][0]) )
    Sample = refine_data(samples)
    data_sets.append(Sample)
    # data_sets = [Sample]
    print(  len(data_sets) )
    print(  len(data_sets[0]) )
    print(  len(data_sets[0][0]) )

# idx = 0
# print([len(a) for a in data_sets ])
# print("data_sets",data_sets)
# for idx in range( len(folder_list) )
for i in range(len(data_sets)):
    ax.fill_between(data_sets[i][0], data_sets[i][2], data_sets[i][3], alpha=0.25, linewidth=0, color=colors[i ])
    # ax.plot(data_sets[0][i][2], data_sets[0][i][3], linewidth=5, color=colors[idx])#, zorder = LineOrder[idx]) #2.5
    ax.plot(data_sets[i][0], data_sets[i][1], linewidth=3, color=colors[i ])#, zorder = LineOrder[idx]) #2.5
    # ax.scatter(data_sets[i][0], data_sets[i][3], s=200, color=colors[idx])
    # ax.scatter(data_sets[0][i][2], data_sets[0][i][3], s=200, color=colors[idx])
# ax.legend(['small loss trick', 'robust classification','vanilla HPO'], fontsize=25)
# ax.legend(['SPT ignore pi(s,a)>80%', 'vanilla SPT HPO'], fontsize=25)
ax.legend([ 'vanilla WCE HPO','WCE HPO with SPT ignore pi(s,a)>0.8 state action pair'], fontsize=12,loc = 'lower right')
scale = 1
ticks = ticker.FuncFormatter(lambda x, pos: '{0:g} '.format(x*scale))
ax.xaxis.set_major_formatter(ticks)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.ylabel('average returns of 100 eval ', fontsize=25)
plt.xlabel('timesteps ', fontsize=25)
# plt.title('NN policy + uniform flipping of advantage signs', fontsize=25)
plt.title(gamename+' full sa with 0.4 uniform flipping of advantage signs  ', fontsize=25)
# plt.set_size_inches(1400,890)
# plt.show()
plt.savefig('./full_sa_spt080_adv040flip_{gamename}.png'.format(gamename = gamename), format='png' )