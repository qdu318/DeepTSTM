import os
import json
import argparse
import torch
import torch.nn as nn

import numpy as np
from utils.data_load import Data_load
from model.DeepTSTM import DeepTSTM
from process.train import Train
from process.evaluate import Evaluate
from model.DPM import DPM as DPM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = json.load(open('./config.json', 'r'))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=201)
parser.add_argument('--weight_file', type=str, default='./saved_weights/')
parser.add_argument('--timesteps_input', type=int, default=12)
parser.add_argument('--timesteps_output', type=int, default=12)
parser.add_argument('--out_channels', type=int, default=1)
parser.add_argument('--spatial_channels', type=int, default=16)
parser.add_argument('--N', type=int, default=50)
parser.add_argument('--features', type=int, default=1)
parser.add_argument('--time_slice', type=list, default=[3,6,12])#default=[1, 2, 3]
parser.add_argument('--dataset', type=list, default=[50, False], help='which dataset(50, 100, 150), Sparse or not('
                                                                      'True or False)')
args = parser.parse_args()


if __name__ == '__main__':
    torch.manual_seed(7)
    W_nodes, data_set = Data_load(args.dataset,config, args.timesteps_input, args.timesteps_output)

    MaxNodeNumber = W_nodes.shape[2]
    model = DeepTSTM(
                num_nodes=args.N,
                out_channels=args.out_channels,
                spatial_channels=args.spatial_channels,
                features=args.features,
                timesteps_input=int(args.timesteps_input),
                timesteps_output=int(args.timesteps_output),
                max_node_number=MaxNodeNumber
            )
    model2 = DPM(
                num_nodes=args.N,
                out_channels=args.out_channels,
                spatial_channels=args.spatial_channels,
                features=args.features,
                timesteps_input=int(args.timesteps_input),
                timesteps_output=int(args.timesteps_output)
            )

    if torch.cuda.is_available():
        model.cuda()
        model2.cuda()
        W_nodes = torch.from_numpy(W_nodes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)#,weight_decay=1e-2
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    L2 = nn.MSELoss()
    L3 = nn.MSELoss()
    RMSE0=[]
    RMSE1=[]
    RMSE2=[]
    MAE0=[]
    MAE1=[]
    MAE2=[]
    MAPE0=[]
    MAPE1=[]
    MAPE2=[]
    Loss0=[]
    Loss1=[]
    Loss2=[]
    Loss_all=[]
    for epoch in range(args.epochs):
        train_loss,trin_loss_ch, X, Y, CHX, CHY = Train(
                        model=model,
                        model2=model2,
                        optimizer=optimizer,
                        optimizer2=optimizer2,
                        loss_meathod=L2,
                        loss_meathod3=L3,
                        W_nodes=W_nodes,
                        data_set=data_set,
                        batch_size=args.batch_size
                    )

        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_loss, eval_index = Evaluate(
                                        model=model,
                                        model2=model2,
                                        loss_meathod=L2,
                                        loss_meathod3=L3,
                                        W_nodes=W_nodes,
                                        time_slice=args.time_slice,
                                        data_set=data_set,
                                        epoch=epoch
                                    )
        print("--------------------------------------------------------------------------------------------------")
        print("epoch: {}/{}".format(epoch, args.epochs))
        print("Training loss: {},{}".format(train_loss,trin_loss_ch))
        for i in range(len(args.time_slice)):
            print("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}"
                  .format(args.time_slice[i] * 5, eval_loss[-(len(args.time_slice) - i)], eval_index['MAE'][-(len(args.time_slice) - i)],
                          eval_index['RMSE'][-(len(args.time_slice) - i)]))#,eval_index['MAPE'][-(len(args.time_slice) - i)]
            if i==0:
                RMSE0.append(eval_index['RMSE'][-(len(args.time_slice) - i)])
                MAE0.append(eval_index['MAE'][-(len(args.time_slice) - i)])
                MAPE0.append(eval_index['MAPE'][-(len(args.time_slice) - i)])
                Loss0.append(eval_loss[-(len(args.time_slice) - i)])
                Loss_all.append(eval_loss[-(len(args.time_slice) - i)])
            if i==1:
                RMSE1.append(eval_index['RMSE'][-(len(args.time_slice) - i)])
                MAE1.append(eval_index['MAE'][-(len(args.time_slice) - i)])
                MAPE1.append(eval_index['MAPE'][-(len(args.time_slice) - i)])
                Loss1.append(eval_loss[-(len(args.time_slice) - i)])
            if i==2:
                RMSE2.append(eval_index['RMSE'][-(len(args.time_slice) - i)])
                MAE2.append(eval_index['MAE'][-(len(args.time_slice) - i)])
                MAPE2.append(eval_index['MAPE'][-(len(args.time_slice) - i)])
                Loss2.append(eval_loss[-(len(args.time_slice) - i)])
        print("---------------------------------------------------------------------------------------------------")

        if not os.path.exists(args.weight_file):
            os.makedirs(args.weight_file)

        if (epoch % 50 == 0) & (epoch != 0):
            torch.save(model, args.weight_file + 'model_' + str(epoch))

    np.savetxt("./results/RMSE0.csv", RMSE0, delimiter=',')
    np.savetxt("./results/RMSE1.csv", RMSE1, delimiter=',')
    np.savetxt("./results/RMSE2.csv", RMSE2, delimiter=',')
    np.savetxt("./results/MAE0.csv", MAE0, delimiter=',')
    np.savetxt("./results/MAE1.csv", MAE1, delimiter=',')
    np.savetxt("./results/MAE2.csv", MAE2, delimiter=',')
    np.savetxt("./results/MAPE0.csv", MAPE0, delimiter=',')
    np.savetxt("./results/MAPE1.csv", MAPE1, delimiter=',')
    np.savetxt("./results/MAPE2.csv", MAPE2, delimiter=',')
    np.savetxt("./results/Loss0.csv", Loss0, delimiter=',')
    np.savetxt("./results/Loss1.csv", Loss1, delimiter=',')
    np.savetxt("./results/Loss2.csv", Loss2, delimiter=',')
    np.savetxt("./results/Loss_all.csv", Loss_all, delimiter=',')

