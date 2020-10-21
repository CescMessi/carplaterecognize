
from findplate.config import opt
import os
import torch as t
from findplate import models
from findplate.data.dataset import MyDataset
from torch.utils.data import DataLoader
from torchnet import meter
from findplate.utils.visualize import Visualizer
from tqdm import tqdm
from torchvision import transforms as T





def write_csv(results,file_name,col1_name,col2_name):
    import csv
    with open(file_name,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([col1_name,col2_name])
        writer.writerows(results)


def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port = opt.vis_port)

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # step2: data
    train_data = MyDataset(opt.train_data_root,train=True)
    val_data = MyDataset(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    # write id and classes into csv file
    data_id_to_class = []
    label_idx = 0
    for label_name in train_data.data_classes:
        data_id_to_class.append([label_idx, label_name])
        label_idx += 1
    print(data_id_to_class)
    id_file_name = opt.id_file
    write_csv(data_id_to_class,id_file_name,'label_idx','label_name')
    
    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)
        
    # step4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(opt.classifier_num)
    previous_loss = 1e10

    # train
    for epoch in range(opt.max_epoch):
        
        loss_meter.reset()
        confusion_matrix.reset()

        for ii,(data,label) in tqdm(enumerate(train_dataloader)):

            # train model 
            input = data.to(opt.device)
            target = label.to(opt.device)


            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()
            
            
            # meters update and visualize
            loss_meter.add(loss.item())
            # detach 一下更安全保险
            confusion_matrix.add(score.detach(), target.detach()) 

            if (ii + 1)%opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])
                
                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()


        model.save()

        # validate and visualize
        val_cm,val_accuracy = val(model,val_dataloader)

        vis.plot('val_accuracy',val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                    epoch = epoch,loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr))
        
        # update learning rate
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        

        previous_loss = loss_meter.value()[0]

@t.no_grad()
def val(model,dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(opt.classifier_num)
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    cm_value_sum = 0
    for i in range(opt.classifier_num):
        cm_value_sum += cm_value[i][i]
    accuracy = 100. * (cm_value_sum) / (cm_value.sum())
    return confusion_matrix, accuracy

def help():
    """
    打印帮助的信息： python file.py help
    """
    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    import fire
    fire.Fire()
