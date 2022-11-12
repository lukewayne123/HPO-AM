from pylab import *
import brewer2mpl
import numpy as np
from matplotlib import pyplot as plt
import csv
import os, sys
import matplotlib.ticker as ticker

# Folder = 'NNslt40flippedduration0'
# root_folder ='sptfullsa040advFlip'
# root_folder ='Cartpole200Epoch'
# root_folder = 'acrobot200epoch'
# root_folder = 'Mountaincar40epoch'
# root_folder ='sptfullsa040advFlipSAindependentAndMoreEpoch'
# root_folder ='sptfullsa045advFlipSAindependentAndMoreEpoch'
# root_folder = 'lunarlander4Mfr20'
# root_folder = 'acrobotfr10sub789'
# root_folder_list = ['lunarlander4Mfr10','lunarlander4Mfr20','lunarlander4Mfr30']
# root_folder_list = ['acrobotfr10sub789','acrobot200epoch','acrobot30']
# root_folder_list = ['acrobotfr10sub789','acrobotfr20re','acrobot30']
# root_folder_list = ['Cartpole200Epoch']
# comapare_folder = ["lunarlander_epsiloncompare"]
# comapare_folder = ["acrobot_epsiloncompare"]
# comapare_folder = ["lunarlander_sptthresholdcompare_fr20_epsilon1"]
# root_folder_list = ["lunarlander_sptthresholdcompare_fr20_epsilon1"]
root_folder_list = ["acrobot_sptthresholdcompare"]
# root_folder_list = ["acrobotfr20reepsilon1","acrobotepsilon2","acrobotepsilon3"]
# root_folder_list = ["lunarlander4Mfr20epsilon1","sptLunarlander20epsilon2","sptLunarlander20epsilon3"]
# gamename = 'cartpole'
# gamename = 'mountaincar'
# gamename = 'Cartpole'
# gamename = 'LunarLander-v2'
# gamename = 'CartPole-v1'
gamename = 'Acrobot-v1'
# gamename = 'MountainCar-v0'

# folder_list = ['vanilla','spt090']
# folder_list = ['vanilla','spt90']
folder_list = ['vanilla','spt90','spt80','spt70']
# folder_list = ['vanilla','spt90','vanilla2','spt902']
merge_flag = True
std_flag = True
# std_flag = False
median_flag = False
# median_flag = True
# folder_list = [gamename+'vanilla',gamename+'spt080',gamename+'slt010dY050']
# folder_list = [gamename+'vanilla',gamename+'spt060']
# folder_list = ['NNslt40flippedduration0','NNrc40flippedduration0','NN0flippedduration0']
# folder_list = ['NNsadependent_slt','NNsadependent_rc','NNsadependent_vanilla']
# seed_list = ['289','666','517','789']
# seed_list = ['1','2','3','4','5','6']
# seed_list = ['1','2','3','4','5']
# seed_list = ['1','2','3' ]
# seed_list = ['1','2','3','4']
# seed_list = ['123','196','285','517','789']
# seed_list = ['123','196','285','517' ] # acrobot
# seed_list = ['123','456','789','8565464','16842464' ] #mountaincar
seed_list = ['123','285','789','78949','16842464' ] #lunarlander 4M fr10
# DataLength = [1] * len(seed_list)
def load_csv(file_path, x_scale = 1):
    print("file_path",file_path)
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
        if median_flag:
            refined_y.append(np.median(ys))
        else:
            refined_y.append(np.mean(ys))
        # min_y.append(np.min(ys))
        # max_y.append(np.max(ys))
        if std_flag:
            min_y.append(np.mean(ys) - np.std(ys))
            max_y.append(np.mean(ys) + np.std(ys))
        else:
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
for root_folder in root_folder_list:
    for idx in range( len(folder_list) ):
        Folder =  folder_list[idx]
        samples = []
        autopath = './{0}/{1}/'.format(root_folder ,Folder)
        # autopath = './{2}/{0}/{1}/'.format(root_folder ,Folder,comapare_folder[0])
        dirs = os.listdir( autopath )
        for x in dirs:
            # samples.append(load_csv("{3}/{0}/{1}/{2}".format(root_folder ,Folder, x,comapare_folder[0] ), x_scale=1))
            samples.append(load_csv("{0}/{1}/{2}".format(root_folder ,Folder, x ), x_scale=1))
        # for x in range( len(seed_list) ):
            # samples.append(load_csv("{0}/{1}/{2}.csv".format(root_folder ,Folder, seed_list[x] ), x_scale=1))
        print( len(samples) )
        print( len(samples[0]) )
        print( len(samples[0][0]) )
        # Sample = refine_data(samples)
        # data_sets.append(Sample)
        if merge_flag:
            samples = refine_data(samples)
        data_sets.append(samples)
        # data_sets = [Sample]
        print(  len(data_sets) )
        print(  len(data_sets[0]) )
        print(  len(data_sets[0][0]) )

# idx = 0
# print([len(a) for a in data_sets ])
# print("data_sets",data_sets)
# for idx in range( len(folder_list) )
# legend_list = [ 'vanilla CE HPO','CE HPO with SPT ignore pi(s,a)>0.9 state action pair' ]
legend_list = [ 'vanilla CE HPO','CE HPO with SPT ignore pi(s,a)>0.9 state action pair','CE HPO with SPT ignore pi(s,a)>0.8 state action pair','CE HPO with SPT ignore pi(s,a)>0.7 state action pair' ]
# legend_list = [ 'vanilla with epsilon=0.1','SPT with epsilon=0.1','vanilla with epsilon=0.2','SPT with epsilon=0.2','vanilla with epsilon=0.3','SPT with epsilon=0.3' ]
# legend_list = ['vanilla with 0.1 uniform flipping of advantage signs','SPT with 0.1 uniform flipping of advantage signs','vanilla with 0.2 uniform flipping of advantage signs','SPT with 0.2 uniform flipping of advantage signs','vanilla with 0.3 uniform flipping of advantage signs','SPT with 0.3 uniform flipping of advantage signs']
for i in range(len(data_sets)):
    if merge_flag:
        if gamename == 'CartPole-v1':
            ax.fill_between(data_sets[i][0], data_sets[i][2], data_sets[i][3], alpha=0.25, linewidth=0, color=colors[i%2 ])
            ax.plot(data_sets[i][0], data_sets[i][1], linewidth=3, color=colors[i%2 ])
        else:
            # ax.fill_between(data_sets[i][0], data_sets[i][2], data_sets[i][3], alpha=0.25, linewidth=0, color=colors[i ])
            # colori = i if i <5 else i+1
            colori = i 
            ax.fill_between(data_sets[i][0], data_sets[i][2], data_sets[i][3], alpha=0.25, linewidth=0, color=colors[ colori ])
            ax.plot(data_sets[i][0], data_sets[i][1], linewidth=3, color=colors[ colori ])
        # ax.plot(data_sets[i][0], data_sets[i][1], linewidth=3, color=colors[i ])
    else:
        # for x in range( len(seed_list) ):
        for x in range( len(data_sets[i]) ):
            # ax.fill_between(data_sets[i][0], data_sets[i][2], data_sets[i][3], alpha=0.25, linewidth=0, color=colors[i ])
            ax.plot(data_sets[i][x][0], data_sets[i][x][1], linewidth=3, color=colors[i ],label=legend_list[i] if x == 0 else "" )#, zorder = LineOrder[idx]) #2.5
    # ax.plot(data_sets[0][i][2], data_sets[0][i][3], linewidth=5, color=colors[idx])#, zorder = LineOrder[idx]) #2.5
    
    # ax.scatter(data_sets[i][0], data_sets[i][3], s=200, color=colors[idx])
    # ax.scatter(data_sets[0][i][2], data_sets[0][i][3], s=200, color=colors[idx])
# ax.legend(['small loss trick', 'robust classification','vanilla HPO'], fontsize=25)
# ax.legend(['SPT ignore pi(s,a)>80%', 'vanilla SPT HPO'], fontsize=25)
# ax.legend([ 'vanilla WCE HPO','WCE HPO with SPT ignore pi(s,a)>0.6 state action pair'], fontsize=12,loc = 'lower right')
# ax.legend([ 'vanilla WCE HPO','WCE HPO with SPT ignore pi(s,a)>0.8 state action pair'], fontsize=12,loc = 'lower right')
if merge_flag:
    # ax.legend([ 'vanilla CE HPO','CE HPO with SPT ignore pi(s,a)>0.9 state action pair' ], fontsize=12,loc = 'lower right')
    ax.legend( legend_list , fontsize=12,loc = 'lower right')
else:
    ax.legend(  fontsize=12,loc = 'lower right')
# ax.legend([ 'vanilla WCE HPO','WCE HPO with SPT ignore pi(s,a)>0.8 state action pair','WCE HPO with SLT 10% deltaY=0.5'], fontsize=12,loc = 'lower right')
scale = 1
ticks = ticker.FuncFormatter(lambda x, pos: '{0:g} '.format(x*scale))
ax.xaxis.set_major_formatter(ticks)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# ax.set_ylim([0, 510]) # for cartpole max rewards
# plt.ylabel('average returns of 100 eval ', fontsize=25)
plt.ylabel('average returns of policy ', fontsize=25)
plt.xlabel('timesteps ', fontsize=25)
# plt.title('NN policy + uniform flipping of advantage signs', fontsize=25)
# plt.title(gamename+' full sa with 0.2 uniform flipping of advantage signs  ', fontsize=25)
# plt.title(gamename+' with 0.2 uniform flipping of advantage signs  ', fontsize=25)
plt.title(gamename+' with threshold1 = 0.9,0.8,0.7 ', fontsize=25)
# plt.title(gamename+' with 10%,20%,30% uniform flipping of advantage signs  ', fontsize=25)
# plt.title(gamename+' full sa with 0.45 uniform flipping of advantage signs  ', fontsize=25)
# plt.set_size_inches(1400,890)
# plt.show()
plt.savefig('./full_sa_spt090_ablation2_{rootfoldername}_{gamename}_merge{mergeflag}_median{medianflag}_std{stdFlag}.png'.format(rootfoldername = root_folder_list[0] ,gamename = gamename, mergeflag = merge_flag, medianflag= median_flag, stdFlag= std_flag ), format='png' )
# plt.savefig('./full_sa_spt090_ablation2_{rootfoldername}_{gamename}_merge{mergeflag}_median{medianflag}_std{stdFlag}.png'.format(rootfoldername = comapare_folder[0] ,gamename = gamename, mergeflag = merge_flag, medianflag= median_flag, stdFlag= std_flag ), format='png' )
# plt.savefig('./full_sa_spt090_200epoch_adv020flip_{gamename}_merge{mergeflag}_median{medianflag}_std{stdFlag}.png'.format(gamename = gamename, mergeflag = merge_flag, medianflag= median_flag, stdFlag= std_flag ), format='png' )
# plt.savefig('./full_sa_spt090_20epoch_adv010flip_{gamename}_merge{mergeflag}_median{medianflag}_std{stdFlag}.png'.format(gamename = gamename, mergeflag = merge_flag, medianflag= median_flag, stdFlag= std_flag ), format='png' )
# plt.savefig('./full_sa_spt090_ablation2_{rootfoldername}_{gamename}_merge{mergeflag}_median{medianflag}_std{stdFlag}.png'.format(rootfoldername = root_folder ,gamename = gamename, mergeflag = merge_flag, medianflag= median_flag, stdFlag= std_flag ), format='png' )
# plt.savefig('./full_sa_spt080_slt010_adv040flip_{gamename}.png'.format(gamename = gamename), format='png' )
# plt.savefig('./full_sa_spt060_adv040flip_moreEpoch_{gamename}.png'.format(gamename = gamename), format='png' )
# plt.savefig('./full_sa_spt060_adv045flip_moreEpoch_{gamename}.png'.format(gamename = gamename), format='png' )
# plt.savefig('./full_sa_spt080_adv040flip_{gamename}.png'.format(gamename = gamename), format='png' )