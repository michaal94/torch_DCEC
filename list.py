
with open('train_list', 'w') as f:
    for dataset in ['bw_separated']:
        for batch in [32, 64]:
            for model in ['CAE_3', 'CAE_3bn', 'CAE_4', 'CAE_4bn', 'CAE_5', 'CAE_5bn']:
                for activation in ['True', 'False']:
                    # for slope in [False, 0.01, 0.2]:
                    for slope in [0.01]:
                        if dataset == 'bw_separated':
                            num_clusters = 16
                        elif dataset == 'all_crops':
                            num_clusters = 12
                        else:
                            num_clusters = 6
                        if not slope:
                            leaky = 'False'
                            slope2 = '0'
                        else:
                            leaky = 'True'
                            slope2 = str(slope)
                        tmp = "/home/michas/anaconda3/envs/tf/bin/python torch_DCEC.py --dataset custom --dataset_path " + dataset + " --pretrain True --batch_size " + str(
                            batch) + " --update_interval 320 --num_clusters " + str(
                            num_clusters) + " --net_architecture " + model + " --activation " + activation + " --leaky " + leaky + " --neg_slope " + slope2
                        # tmp = "/home/michas/anaconda3/envs/tf/bin/python torch_DCEC.py --dataset custom --dataset_path " + dataset + " --pretrain True --batch_size " + str(
                        #     1) + " --update_interval 320 --num_clusters " + str(
                        #     num_clusters) + " --net_architecture " + model + " --activation " + activation + " --leaky " + leaky + " --neg_slope " + slope2 + ' --epochs 1 --epochs_pretrain 1'
                        f.write(tmp + '\n')
