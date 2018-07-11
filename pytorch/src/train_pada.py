import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
import torch.utils.data as util_data
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random

optim_dict = {"SGD": optim.SGD}

def image_classification_predict(loader, model, test_10crop=True, gpu=True, softmax_param=1.0):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test'+str(i)]) for i in range(10)]
        for i in range(len(loader['test0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = Variable(inputs[j].cuda())
                labels = Variable(labels.cuda())
            else:
                for j in range(10):
                    inputs[j] = Variable(inputs[j])
                labels = Variable(labels)
            outputs = []
            for j in range(9):
                _, predict_out = model(inputs[j])
                outputs.append(nn.Softmax(dim=1)(softmax_param * predict_out))
            _, outputs_center = model(inputs[9])
            outputs.append(nn.Softmax(dim=1)(softmax_param * outputs_center))
            softmax_outputs = sum(outputs)
            outputs = outputs_center
            if start_test:
                all_output = outputs.data.float()
                all_softmax_output = softmax_outputs.data.cpu().float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        iter_val = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_val.next()
            inputs = data[0]
            if gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            _, outputs = model(inputs)
            softmax_outputs = nn.Softmax(dim=1)(softmax_param * outputs)
            if start_test:
                all_output = outputs.data.cpu().float()
                all_softmax_output = softmax_outputs.data.cpu().float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    _, predict = torch.max(all_output, 1)
    return all_softmax_output, predict, all_output, all_label

def image_classification_test(loader, model, test_10crop=True, gpu=True, iter_num=-1):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test'+str(i)]) for i in range(10)]
        for i in range(len(loader['test0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = Variable(inputs[j].cuda())
                labels = Variable(labels.cuda())
            else:
                for j in range(10):
                    inputs[j] = Variable(inputs[j])
                labels = Variable(labels)
            outputs = []
            for j in range(10):
                _, predict_out = model(inputs[j])
                outputs.append(nn.Softmax(dim=1)(predict_out))
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)       
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label) / float(all_label.size()[0])
    return accuracy


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
    prep_dict["target"] = prep.image_train( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
    else:
        prep_dict["test"] = prep.image_test( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
               
    ## set loss
    class_criterion = nn.CrossEntropyLoss()
    transfer_criterion = loss.PADA
    loss_params = config["loss"]

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = util_data.DataLoader(dsets["source"], \
            batch_size=data_config["source"]["batch_size"], \
            shuffle=True, num_workers=4)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = util_data.DataLoader(dsets["target"], \
            batch_size=data_config["target"]["batch_size"], \
            shuffle=True, num_workers=4)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"+str(i)] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"]["val"+str(i)])
            dset_loaders["test"+str(i)] = util_data.DataLoader(dsets["test"+str(i)], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=4)

            dsets["target"+str(i)] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["test"]["val"+str(i)])
            dset_loaders["target"+str(i)] = util_data.DataLoader(dsets["target"+str(i)], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=4)
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=4)

        dsets["target_test"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["target_test"] = MyDataLoader(dsets["target_test"], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=4)

    class_num = config["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = network.network_dict[net_config["name"]](use_bottleneck=net_config["use_bottleneck"], new_cls=net_config["new_cls"], class_num=class_num)


    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()

    ## collect parameters
    if net_config["new_cls"]:
        if net_config["use_bottleneck"]:
            parameter_list = [{"params":base_network.feature_layers.parameters(), "lr":1}, \
                            {"params":base_network.bottleneck.parameters(), "lr":10}, \
                            {"params":base_network.fc.parameters(), "lr":10}]
        else:
            parameter_list = [{"params":base_network.feature_layers.parameters(), "lr":1}, \
                            {"params":base_network.fc.parameters(), "lr":10}]
    else:
        parameter_list = [{"params":base_network.parameters(), "lr":1}]

    ## add additional network for some methods
    class_weight = torch.from_numpy(np.array([1.0] * class_num))
    if use_gpu:
        class_weight = class_weight.cuda()
    ad_net = network.AdversarialNetwork(base_network.output_num())
    gradient_reverse_layer = network.AdversarialLayer()
    if use_gpu:
        ad_net = ad_net.cuda()
    parameter_list.append({"params":ad_net.parameters(), "lr":10})
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]


    ## train   
    len_train_source = len(dset_loaders["source"]) - 1
    len_train_target = len(dset_loaders["target"]) - 1
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == 0:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, \
                base_network, test_10crop=prep_config["test_10crop"], \
                gpu=use_gpu)
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str)
            config["out_file"].flush()
            print(log_str)
        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                "iter_{:05d}_model.pth.tar".format(i)))
                    
       
        if (i % loss_params["update_iter"] == loss_params["update_iter"] - 1) and (i > loss_params["start_iter"]):
            base_network.train(False)
            target_fc8_out, _, target_before_softmax, target_label = image_classification_predict(dset_loaders, base_network, softmax_param=1.0)
            class_weight = torch.mean(target_fc8_out, 0)
            class_weight = (class_weight / torch.mean(class_weight)).cuda().view(-1)
            class_criterion = nn.CrossEntropyLoss(weight = class_weight)
        

        ## train one iter
        base_network.train(True)
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        if use_gpu:
            inputs_source, inputs_target, labels_source = \
                Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), \
                Variable(labels_source).cuda()
        else:
            inputs_source, inputs_target, labels_source = Variable(inputs_source), \
                Variable(inputs_target), Variable(labels_source)
           
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        features, outputs = base_network(inputs)

        softmax_out = nn.Softmax(dim=1)(outputs).detach()
        ad_net.train(True)
        weight_ad = torch.zeros(inputs.size(0))
        label_numpy = labels_source.data.cpu().numpy()
        for j in range(inputs.size(0) / 2):
            weight_ad[j] = class_weight[int(label_numpy[j])]
        weight_ad = weight_ad / torch.max(weight_ad[0:inputs.size(0)/2])
        for j in range(inputs.size(0) / 2, inputs.size(0)):
            weight_ad[j] = 1.0            
        transfer_loss = transfer_criterion(features, ad_net, gradient_reverse_layer, \
                                           weight_ad, use_gpu)

        classifier_loss = class_criterion(outputs.narrow(0, 0, inputs.size(0)/2), labels_source)

        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('gpu_id', type=str, nargs='?', default='0', help="device id to run")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

    # train config
    
    config = {}
    config["num_iterations"] = 50004
    config["test_interval"] = 500
    config["snapshot_interval"] = 10000
    config["output_for_test"] = True
    config["output_path"] = "../snapshot/no"
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":True, "resize_size":256, "crop_size":224}
    config["loss"] = {"trade_off":1.0, "update_iter":500, "start_iter":-1}
    config["network"] = {"name":"ResNet50", "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True}
    config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75} }

    config["dataset"] = "office"
    if config["dataset"] == "office":
        config["data"] = {"source":{"list_path":"../data/office/dslr_31_list.txt", "batch_size":36}, \
                          "target":{"list_path":"../data/office/amazon_10_list.txt", "batch_size":36}, \
                          "test":{"list_path":"../data/office/amazon_10_list.txt", "batch_size":4}}
        if "amazon" in config["data"]["test"]["list_path"]:
            config["optimizer"]["lr_param"]["init_lr"] = 0.0003
        else:
            config["optimizer"]["lr_param"]["init_lr"] = 0.001
        config["loss"]["update_iter"] = 500
        config["class_num"] = 31
    print(train(config))
