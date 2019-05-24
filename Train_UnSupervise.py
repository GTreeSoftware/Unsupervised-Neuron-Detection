from Common import *
from GetModel import *
from utils.tools import *
from utils.vis_tool import Visualizer
from config import opt
from utils.loss import DiceLossPlusCrossEntrophy
from utils.meters import AverageMeter
import torch.nn as nn
import torch
from tqdm import tqdm
import time
from utils.eval_metric import ConfusionMeter
from dataset.joint_transform import *
from Validation import val_model


###### used for training for one epoch ###################
def TrainOneEpoch(train_loader,model,optimizer,criterion,epoch_num,vis_tool,record_value):
    losses = AverageMeter()
    train_eval = ConfusionMeter(num_class=opt.out_dim)

    # calculate the final result
    train_dice=AverageMeter()
    train_recall=AverageMeter()

    best_value=record_value

    model.train()

    for batch_ids, (data, target) in enumerate(train_loader):
        if opt.use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model(data)

        # calculate the weight of the batch:
        weight = GetWeight(opt,target, slr=0,is_t=0)
        loss = criterion(output, target,weight=weight)

        loss.backward()
        optimizer.step()

        # update the loss value
        losses.update(loss.item())

        # calculate the metrics for evaluation:
        _, pred = torch.max(output, 1)
        train_eval.update(pred, target)

        avg_loss=losses.avg
        dice_value=train_eval.get_scores('Dice')
        recall_value=train_eval.get_scores('Recall')

        train_dice.update(dice_value)
        train_recall.update(recall_value)

        # for visualization
        if batch_ids % opt.train_plotfreq == 0:
            vis_tool.plot('Train_Loss', loss.item())
            vis_tool.plot('Train_Dice', dice_value)
            vis_tool.plot('Train_Recall', recall_value)

        print('Train:Batch_Num:{}  Loss:{:.3f}  Dice:{:.3f}  Recall:{:.3f}'.format(batch_ids,loss.item(),dice_value,recall_value))


    return avg_loss,train_dice.avg,train_recall.avg,best_value



# define the main function
def main():
    # print the parameters:
    opt._parse()

    train_aug=JointCompose([JointRandomFlip(),
                            JointRandomRotation(),
                            JointRandomGaussianNoise(8),
                            JointRandomSubTractGaussianNoise(8),
                            JointRandomBrightness([-0.3,0.3]),
                            JointRandomIntensityChange([-0.3,0.3]),
                            ])


    # load the dataloader
    prefix='Unsupervised_Train'
    train_loader = GetSemiDatasetLoader(opt, prefix, phase='Train', phase_1='train', augument=train_aug)
    val_loader=GetSemiDatasetLoader(opt,prefix,phase= 'Val', phase_1='val')

    # get the net work
    model=GetModel(opt)

    # if opt.load_state:
    #     ##### fine tune the result ####
    #     ## replace your model
    #     parameters_name = 'XXXXXXXXXXX'
    #
    #     model_CKPT = torch.load(parameters_name)
    #     model.load_state_dict(model_CKPT['state_dict'])

    optimizer=GetOptimizer(opt,model)
    scheduler=GetScheduler(opt,optimizer)

    # get the loss function
    criterion=DiceLossPlusCrossEntrophy()


    vis_tool=Visualizer(env='VoxResNet_3D')
    log=WriteLog(opt)
    log.write('epoch |train_loss |train_dice |train_recall |valid_loss |valid dice |valid_recall |time          \n')
    log.write('------------------------------------------------------------\n')

    # record the value : for better saving:dice,recall
    record_value=np.array([0,0])

    # begin to train:
    for epoch_num in range(opt.train_epoch):
        start_time=time.time()
        scheduler.step()

        avg_loss,train_dice,train_recall,best_value=TrainOneEpoch(train_loader, model, optimizer, criterion, epoch_num, vis_tool,record_value)
        record_value=best_value

        # the information
        run_time = time.time()-start_time
        print('Train Epoch{} run time is:{:.3f}m and {:.3f}s'.format(epoch_num,run_time//60,run_time%60))
        print('Loss:{:.3f}  Recall:{:.3f}  Dice:{:.3f}'.format(avg_loss,train_recall,train_dice))


        # begin to save the parameters
        save_name1 = (opt.save_parameters_name+'_epoch_{}').format(epoch_num)

        state={'epoch': epoch_num + 1,
               'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               }

        save_parameters(state, save_name1)

        # using val run to validate the dataset
        if opt.val_run:
            val_avgloss,val_dice,val_recall=val_model(opt,val_loader,model,criterion,vis_tool)

            print('Test Loss:{:.3f}  Recall:{:.3f}  Dice:{:.3f}'.format(val_avgloss,val_recall,val_dice))

        log.write('%d |%0.3f |%0.3f |%0.3f |%0.3f |%0.3f |%0.3f |%0.3f \n' % (epoch_num,avg_loss,train_dice,train_recall,
                                                                    val_avgloss,val_dice,val_recall,run_time))
        log.write('\n')




if __name__=="__main__":
    main()



















































