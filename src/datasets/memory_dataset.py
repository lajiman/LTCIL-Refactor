import random
import numpy as np
from PIL import Image
from numpy.core.shape_base import block
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MemoryDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all images in memory"""

    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # x = Image.fromarray(self.images[index])
        # x = self.transform(x)
        # y = self.labels[index]
        # return x, y
        x = self.images[index]
        y = self.labels[index]

        # Case 1: AV exemplar, e.g. x = (visual_feat, audio_feat)
        if isinstance(x, tuple):
            # keep feature tensors/arrays as-is; transform usually image-only, so skip it
            return x, y

        # Case 2: image exemplar (original behavior)
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y
        

def get_data(trn_data, tst_data, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None):
    """Prepare data: dataset splits, task partition, class order"""
    '''
    trn_data and tst_data are dictionaries with keys 'x' and 'y' for images and labels respectively.
    {'x': [image1, image2, ...], 'y': [label1, label2, ...]}
    '''
    data = {}
    taskcla = []
    clsanalysis = {}
    testclsanalysis={}
    list_of_clsnum=[]
    inorder = True
    
    # in acvil, the class order is determined by the order of the classes in the dataset, which is fixed. 
    # in that case, we don't need to shuffle the classes, and we can directly use the class order in the dataset.
    if class_order is None:
        num_classes = len(np.unique(trn_data['y']))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    # example: 
    # cpertask = [10, 10, 10], cpertask_cumsum = [10, 20, 30], init_class = [0, 10, 20]
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}
        clsanalysis[tt] = np.zeros(cpertask[tt])    # for analysis of class distribution in training set.
        testclsanalysis[tt] = np.zeros(cpertask[tt])

    #training set analysis
     
    # labelcounter=np.zeros(num_classes)
    # ALL OR TRAIN
    # Filters the training data to only include classes in the class_order list. If there are any classes in trn_data['y'] that are not in class_order, they will be removed from trn_data['x'] and trn_data['y'].
    filtering = np.isin(trn_data['y'], class_order)
    if filtering.sum() != len(trn_data['y']):
        trn_data['x'] = trn_data['x'][filtering]
        trn_data['y'] = np.array(trn_data['y'])[filtering]
    
    # print(order)
    for this_image, this_label in zip(trn_data['x'], trn_data['y']):
        # If shuffling is false, it won't change the class number
        # example, if class_order = [8, 1, 3, 7, ...], then 8 -> 0, 1 -> 1, 3 -> 2, 7 -> 3, ...
        # which means we can sample the vggsounds in an elegent way.
        this_label = class_order.index(this_label)
        # add it to the corresponding split
        # example. if this_label = 12, and cpertask_cumsum = [10, 20, 30], then this_task = 12, this_task = (True, False, False).sum() = 1, which means this_label belongs to task 1 (the second task).
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(this_image)
        # for each task, labels start from 0
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])
        clsanalysis[this_task][this_label - init_class[this_task]]+=1
    

    # ALL OR TEST
    filtering = np.isin(tst_data['y'], class_order)
    if filtering.sum() != len(tst_data['y']):
        tst_data['x'] = tst_data['x'][filtering]
        tst_data['y'] = tst_data['y'][filtering]
    for this_image, this_label in zip(tst_data['x'], tst_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)
        # this_label = order.index(this_label)
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])
        testclsanalysis[this_task][this_label - init_class[this_task]]+=1
    # clsanalyze(clsanalysis,testclsanalysis)
    # check classes, if the number of classes in each task is correct, and if the class distribution is correct.
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    # in avcil, the validation set is fixed to a certain number, 50. When modifying the code, we should add arguments to choose whether to use a fixed number(and how much) of samples for validation or a ratio of the training set.
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0]) # get the indices of the samples in the training set that belong to class cc in task tt
                if int(np.round(len(cls_idx) * validation))==0:
                    rnd_img=[0]
                else:
                    rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation))) # randomly sample some of these indices to be in the validation set, the number of samples is determined by the validation ratio. for example, if there are 100 samples in class cc, and validation ratio is 0.2, then we will randomly sample 20 indices from cls_idx to be in the validation set.
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][rnd_img[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])

    # convert them to numpy arrays
    for tt in data.keys():
        for split in ['trn', 'val', 'tst']:
            data[tt][split]['x'] = np.asarray(data[tt][split]['x'])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))    # example, taskcla = [(0, 10), (1, 10), (2, 10)], which means task 0 has 10 classes, task 1 has 10 classes, task 2 has 10 classes.
        n += data[t]['ncla']
    data['ncla'] = n    # example, ncla = 30, which means there are 30 classes in total.

    # example: data = {0: {'name': 'task-0', 'trn': {'x': [image1, image2, ...], 'y': [label1, label2, ...]}, 'val': {'x': [image1, image2, ...], 'y': [label1, label2, ...]}, 'tst': {'x': [image1, image2, ...], 'y': [label1, label2, ...]}, 'ncla': 10}, 1: {'name': 'task-1', 'trn': {'x': [image1, image2, ...], 'y': [label1, label2, ...]}, 'val': {'x': [image1, image2, ...], 'y': [label1, label2, ...]}, 'tst': {'x': [image1, image2, ...], 'y': [label1, label2, ...]}, 'ncla': 10}, 2: {'name': 'task-2', 'trn': {'x': [image1, image2, ...], 'y': [label1, label2, ...]}, 'val': {'x': [image1, image2, ...], 'y': [label1, label2, ...]}, 'tst': {'x': [image1, image2, ...], 'y': [label1, label2, ...]}, 'ncla': 10}, 'ncla': 30}, which means there are 3 tasks in total. Task 0 has 10 classes and the training set of task 0 contains some images and labels of these 10 classes. Task 1 has 10 classes and the training set of task 1 contains some images and labels of these 10 classes. Task 2 has 10 classes and the training set of task 2 contains some images and labels of these 10 classes. In total there are 30 classes.
    # example: taskcla = [(0, 10), (1, 10), (2, 10)], which means task 0 has 10 classes, task 1 has 10 classes, task 2 has 10 classes.
    # example: class_order = [8, 1, 3, 7, ...], which means the original class 8 is now class 0, the original class 1 is now class 1, the original class 3 is now class 2, the original class 7 is now class 3, ...
    return data, taskcla, class_order


