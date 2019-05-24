from utils.meters import AverageMeter,AverageMeterSet
import torch.nn as nn
import torch
import time
from utils.eval_metric import ConfusionMeter
from dataset.joint_transform import *


def val_model(opt,val_loader,model,criterion,vis_tool,name1='1'):
    # begin to test the dataset
    model.eval()

    val_eval=ConfusionMeter(num_class=opt.out_dim)

    # meters=AverageMeterSet()
    # calculate the average values
    val_dice=AverageMeter()
    val_loss=AverageMeter()
    val_recall=AverageMeter()

    for batch_ids,(data,target) in enumerate(val_loader):
        if opt.use_cuda:
            data,target=data.cuda(),target.cuda()

            output=model(data)

            with torch.no_grad():
                loss=criterion(output,target)
                _,pred=torch.max(output,dim=1)

                val_loss.update(loss.item())
                val_eval.update(pred,target)

                avg_loss = val_loss.avg
                dice_value = val_eval.get_scores('Dice')
                recall_value = val_eval.get_scores('Recall')

                val_recall.update(recall_value)
                val_dice.update(dice_value)

                # begin to play
                if batch_ids % opt.val_plotfreq == 0:
                    vis_tool.plot('Val_Loss'+name1, loss.item())
                    vis_tool.plot('Val_Dice'+name1, dice_value)
                    vis_tool.plot('Val_Recall'+name1, recall_value)

                print('Val: Batch_Num:{}  Loss:{:.3f}  Dice:{:.3f}  Recall:{:.3f}'.format(batch_ids, loss.item(),
                                                                                     dice_value, recall_value))


    return avg_loss, val_dice.avg,val_recall.avg