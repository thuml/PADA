# PADA
PyTorch implementation for [Partial Adversarial Domain Adaptation (ECCV 2018)]

## Prerequisites
Linux or OSX

NVIDIA GPU + CUDA (may CuDNN) and corresponding PyTorch framework (version 0.3.1)

Python 2.7/3.5

## Datasets
We use Office-31, Office-Home and ImageNet-Caltech dataset in our experiments. We use [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt), [Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256) and [ImageNet-2012](http://www.image-net.org) datasets. The ImageNet-Caltech dataset will be published soon. 

The lists of dataset are in [data](./data) directory. The "data/imagenet-caltech/imagenet_1000_list.txt" is too large, we put it on the [google drive](https://drive.google.com/open?id=1QARHJoxVpyB2EQZyrBbBHSiEQjBowPD2). 

For Office-31 dataset, "name_31_list.txt"(name="amazon", "webcam", "dslr") is the source list file and "name_10_list.txt" is the target list file.

For Office-Home dataset, "name.txt"(name="Art", "Clipart", "Product", "Real_World") is the source list file and "name_shared.txt" is the target list file.

For ImageNet-Caltech dataset, "imagenet_1000_list.txt" is the source file list for task "I->C" and "caltech_84_list.txt" is the target file list. "caltech_256_list.txt" is the source file list for task "C->I" and "imagenet_val_84_list.txt" is the target file list.

You can also modify the list file(txt format) in ./data as you like. Each line in the list file follows the following format:
```
<image path><space><label representation>
```

## Training and Evaluation
First, you can manually download the PyTorch pre-trained model introduced in `torchvision' library or if you have connected to the Internet, you can automatically downloaded them.
Then, you can train the model for each dataset using the followling command.
```
cd src
python train_pada.py --gpu_id 2 --net ResNet50 --dset office --s_dset_path ../data/office/webcam_31_list.txt --t_dset_path ../data/office/amazon_10_list.txt --test_interval 500 --snapshot_interval 5000 --output_dir san
```
You can set the command parameters to switch between different experiments. 
- "gpu_id" is the GPU ID to run experiments.
- "dset" parameter is the dataset selection. In our experiments, it can be "office" (for all the Office-31 tasks), "office-home" (for all the Office-Home tasks), "imagenet" (for task ImageNet->Caltech) and "caltech" (for Caltech->ImageNet).
- "s_dset_path" is the source dataset list.
- "t_dset_path" is the target dataset list.
- "test_interval" is the interval of iterations between two test phase.
- "snapshot_interval" is the interval of iterations between two snapshot models.
- "output_dir" is the output directory of the log and snapshot.
- "net" sets the base network. For details of setting, you can see network.py.
    - For AlexNet, "net" is AlexNet.
    - For VGG, "net" is like VGG16. Detail names are in network.py.
    - For ResNet, "net" is like ResNet50. Detail names are in network.py.
