import os
import numpy as np
import math

from tqdm import tqdm
from torch.utils.data import Dataset


class KITTIINSTANCEDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=1024, sample_rate=0.3, transform=None):
        super().__init__()
        self.num_point = num_point
        self.transform = transform
        self.sample_rate = sample_rate
        if split == 'train':
            with open(os.path.join(data_root, 'ImageSets', 'train.txt'), 'r') as f:
                data_str = f.read()
                set_split = data_str.split('\n')
                set_split.sort()
        else:
            with open(os.path.join(data_root, 'ImageSets', 'val.txt'), 'r') as f:
                data_str = f.read()
                set_split = data_str.split('\n')
                set_split.sort()

        self.collect_points, self.collect_labels = [], []
        self.collect_coord_min, self.collect_coord_max = [], []
        num_point_all = []
        labelweights = np.ones(2)

        for data_name in tqdm(set_split, total=len(set_split)):
            point_path = os.path.join(data_root, 'training', 'point_feature', '{}_point_feature.npy'.format(data_name))
            label_path = os.path.join(data_root, 'training', 'point_sem_label', '{}_sem_label.npy'.format(data_name))
            instance_mask_path = os.path.join(data_root, 'training', 'point_ins_mask', '{}_ins_mask.npy'.format(data_name))
            point = np.load(point_path)   # (x, y, z, prob, r, g, b)  N*7
            label = np.load(label_path)
            instance_mask = np.load(instance_mask_path)
            tmp, _ = np.histogram(label, range(3))
            labelweights += tmp
            for im in range(instance_mask.shape[1]):
                ins_mask = instance_mask[:, im]
                ins_point = point[ins_mask, :]
                ins_label = label[ins_mask]
                if ins_point.size > 0:
                    coord_min, coord_max = np.amin(np.abs(ins_point), axis=0)[:3], np.amax(np.abs(ins_point), axis=0)[:3]
                    self.collect_points.append(ins_point), self.collect_labels.append(ins_label)
                    self.collect_coord_min.append(coord_min), self.collect_coord_max.append(coord_max)
                    num_point_all.append(ins_label.size)
                    # print(ins_label.size)

        print('mean num {}'.format(np.mean(num_point_all)))
        print('max num {}'.format(np.max(num_point_all)))
        print('min num {}'.format(np.min(num_point_all)))
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        data_idxs = range(len(self.collect_labels))
        self.data_idxs = np.array(data_idxs)
        print("Totally {} samples in {} set.".format(len(self.data_idxs), split))

    def __getitem__(self, idx):
        data_idx = self.data_idxs[idx]
        point = self.collect_points[idx]   # N * 7
        label = self.collect_labels[idx]   # N
        N_point = point.shape[0]

        if N_point < self.num_point:
            sample_num = math.floor(N_point*self.sample_rate)
            sample_idx = np.random.choice(range(N_point), sample_num)
            sample_point = point[sample_idx, :]
            sample_label = label[sample_idx]
            expand_num = math.ceil((self.num_point-N_point)/sample_num)

            # 将各个维度均重复
            expand_point = np.tile(sample_point, (expand_num, 1))
            expand_label = np.tile(sample_label, expand_num)
            expand_point = np.reshape(expand_point, (-1, 7))
            expand_label = np.reshape(expand_label, (-1,))
            expand_point[:, :3] = expand_point[:, :3] + np.random.rand(expand_point.shape[0], 3) * 0.1
            point = np.concatenate((point, expand_point), axis=0)
            label = np.concatenate((label, expand_label))

        if point.shape[0] > self.num_point*2:
            while (True):
                center = point[np.random.choice(N_point)][:3]
                coord_x, coord_y, _ = self.collect_coord_max[idx] - self.collect_coord_min[idx]
                block_min = center - [coord_x / 2.0, coord_y / 2.0, 0]
                block_max = center + [coord_x / 2.0, coord_y / 2.0, 0]
                point_idxs = np.where((point[:, 0] >= block_min[0]) & (point[:, 0] <= block_max[0]) & (point[:, 1] >= block_min[1]) & (point[:, 1] <= block_max[1]))[0]
                if point_idxs.size > 1024:
                    break
        else:
            center = point[np.random.choice(N_point)][:3]
            point_idxs = np.arange(point.shape[0])

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = point[selected_point_idxs, :]  # num_point * 7
        current_points = np.zeros((self.num_point, 10))  # num_point * 10
        current_points[:, 7] = selected_points[:, 0] / self.collect_coord_max[idx][0]
        current_points[:, 8] = selected_points[:, 1] / self.collect_coord_max[idx][1]
        current_points[:, 9] = selected_points[:, 2] / self.collect_coord_max[idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 4:7] /= 255.0
        current_points[:, 0:7] = selected_points
        current_labels = label[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.data_idxs)

class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(13)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(14))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3] # 获取点云的空间范围，xyz上的最小最大值
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)   # 根据block_size对点云在x上的划分
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)   # 根据block_size对点云在y上的划分
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size

                # 根据划分区域获取对应的点云索引
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))   # 将每个grid中的点云再分配成固定数量的block中，num_batch就是要分配的block数量
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]   # 对点云的xyz进行归一化处理
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0 # 对点云颜色进行归一化处理
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))  # [block_num, block_points, point_feature] 根据block划分点云
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    # KITTIINSTANCEDataset
    # '''
    data_root = '/data/szy4017/code/Pointnet_Pointnet2_pytorch/data/kitti_instance'
    num_point, test_area, block_size, sample_rate = 1024, 5, 1.0, 0.3

    point_data = KITTIINSTANCEDataset(split='train', data_root=data_root, num_point=num_point, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()    
    # '''

    # ScannetDatasetWholeScene
    '''
    data_root = '/data/szy4017/code/Pointnet_Pointnet2_pytorch/data/s3dis/stanford_indoor3d/'
    point_data = ScannetDatasetWholeScene(data_root, split='test', test_area=5, block_points=4096)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)

    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()    
    '''
