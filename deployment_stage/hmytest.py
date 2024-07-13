import os, sys
from pprint import PrettyPrinter
import pickle
import numpy as np
import random
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# from deployment_stage.collect_all_data import collect_all_data
from offline_stage_2.utils import cal_agent_oppo_obs_mean
from offline_stage_2.utils import load_agent_oppo_data
from offline_stage_2.config import Config, get_config_dict
from deployment_stage.collect_all_data import collect_all_data
from deployment_stage.collect_data import collect_data

def main():
    
    # data = collect_all_data()
    # with open('collected_data_target1.0_pel0710014138_all_window.pkl', 'wb') as f:
    #     pickle.dump(data, f)

    with open(('collected_data_target1.0_pel0710014138_all_window.pkl'), 'rb') as f:
        online_data = pickle.load(f)    
    
    num_all = len(online_data[0])
    print(num_all)
    sum = np.zeros(num_all)
    list_up = [[]]
    for k in range(num_all):
        for i in range(len(online_data[0][k][0]["rewards"])):
            sum[k] = sum[k] + online_data[0][k][0]["rewards"][i]
    num_up = 0
    all_up = 0
    num_up_list = []
    for j in range(num_all):
        if(sum[j]>0.75):
            list_up[0].append(online_data[0][j])
            num_up_list.append(j)
            num_up = num_up + 1
    print(num_up)
    print(num_up_list)
    # for i in range(num_up):
    #     all_up = all_up + sum[num_up_list[i]]
    # avg = all_up / num_up
    # print(avg)
    # batch_inds_up = np.random.choice(
    #     np.arange(num_up),
    #     size=128,
    #     replace=False,
    # )
    # batch_inds = []
    # for i in range(len(batch_inds_up)):
    #     batch_inds.append(num_up_list[batch_inds_up[i]])
    # random.shuffle(batch_inds)
    # print(batch_inds)
    # online_data = collect_all_data()
    # with open('collected_data_target80_400_peloffline_2.pkl', 'wb') as f:
    #     pickle.dump(online_data, f)
    # with open(('/home/khuang@kean.edu/hmy/iclr2024-TAO/deployment_stage/collected_data_target80_400.pkl'), 'rb') as f:
    #     data = pickle.load(f)
    # sum = np.zeros(800)
    # list_up = [[]]
    # list_down = [[]]
    # for k in range(800):
    #     for i in range(100):
    #         sum[k] = sum[k] + data[0][k][0]["rewards"][i]
    # num_up = 0
    # num_up_list = []
    # num_down = 0
    # num_down_list = []
    # all_up = 0 
    # all_down = 0
    # for j in range(800):
    #     if(sum[j]>40):
    #         list_up[0].append(data[0][j])
    #         num_up_list.append(j)
    #         num_up = num_up + 1
    #     else :
    #         list_down[0].append(data[0][j])
    #         num_down_list.append(j)
    #         num_down = num_down + 1
    
    # print(num_up)
    # print(num_up_list)
    # for i in range(num_up):
    #     all_up = all_up + sum[num_up_list[i]]
    # print(all_up)
    # print(num_down)
    # print(num_down_list)
    # for j in range(num_down):
    #     all_down = all_down + sum[num_down_list[j]]
    # print(all_down)

    # rate = (num_up*4)/(num_up*4+num_down)     # 好的数据多训练
    # batch_inds_up = np.random.choice(
    #     np.arange(num_up),
    #     size=(int)(128*rate),
    #     replace=False,
    # )
    # batch_inds_down = np.random.choice(
    #     np.arange(num_down),
    #     size=128-(int)(128*rate),
    #     replace=False,        
    # )
    # print(len(batch_inds_up))
    # print(len(batch_inds_down))
    # batch_inds = []
    # for i in range(len(batch_inds_up)):
    #     batch_inds.append(num_up_list[batch_inds_up[i]])
    # for j in range(len(batch_inds_down)):
    #     batch_inds.append(num_down_list[batch_inds_down[j]])
    # print(batch_inds)
    # random.shuffle(batch_inds)
    # print(batch_inds) # sum = np.zeros(400)
    # for j in range(400):
    #     for i in range(100):
    #         sum[j] = sum[j] + data[0][j][0]["rewards"][i]

if __name__ == '__main__':
    main()

# with open(('/home/khuang@kean.edu/hmy/iclr2024-TAO/deployment_stage/collected_data_target80.pkl'), 'rb') as f:
    #     data = pickle.load(f)
    # list_up = [[]]
    # list_down = [[]]
    # for k in range(400):
    #     for i in range(100):
    #         sum[k] = sum[k] + data[0][k][0]["rewards"][i]
    # num_up = 0
    # num_up_list = []
    # num_down = 0
    # num_down_list = []
    # for j in range(400):
    #     if(sum[j]>40):
    #         list_up[0].append(data[0][j])
    #         num_up_list.append(j)
    #         num_up = num_up + 1
    #     else :
    #         list_down[0].append(data[0][j])
    #         num_down_list.append(j)
    #         num_down = num_down + 1
    
    # print(num_up)
    # print(num_up_list)
    # print(num_down)
    # print(num_down_list)
    # rate = (num_up*4)/(num_up*4+num_down)     # 好的数据多训练
    # batch_inds_up = np.random.choice(
    #     np.arange(num_up),
    #     size=(int)(128*rate),
    #     replace=False,
    # )
    # batch_inds_down = np.random.choice(
    #     np.arange(num_down),
    #     size=128-(int)(128*rate),
    #     replace=False,        
    # )
    # print(len(batch_inds_up))
    # print(len(batch_inds_down))
    # batch_inds = []
    # for i in range(len(batch_inds_up)):
    #     batch_inds.append(num_up_list[batch_inds_up[i]])
    # for j in range(len(batch_inds_down)):
    #     batch_inds.append(num_down_list[batch_inds_down[j]])
    # print(batch_inds)
    # random.shuffle(batch_inds)
    # print(batch_inds) # sum = np.zeros(400)
    # for j in range(400):
    #     for i in range(100):
    #         sum[j] = sum[j] + data[0][j][0]["rewards"][i]
    # print(len(sum))
    # print(np.max(sum))
    # print(np.min(sum))
    # print(len(data))
    # print(len(data[0]))
    # print(len(data[0][0]))
    # print(len(data[0][0][0]))
    # print(len(data[0][0][0]["rewards"]))
    # sum = np.zeros(400)
    
