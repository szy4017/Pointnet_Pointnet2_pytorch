import os
import numpy as np
import math

from tqdm import tqdm
from torch.utils.data import Dataset


class KITTIINSTANCEDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=1024, sample_rate=1.0, transform=None):
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

class KittiInstanceDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, data_root, block_points=1024, split='test', sample_rate=0.3):
        self.block_points = block_points
        self.root = data_root
        self.split = split
        self.sample_rate = sample_rate
        self.scene_points_num = []
        assert split in ['train', 'test']
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
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.instance_indices_list = []
        labelweights = np.ones(2)

        for data_name in tqdm(set_split, total=len(set_split)):
            point_path = os.path.join(data_root, 'training', 'point_feature', '{}_point_feature.npy'.format(data_name))
            label_path = os.path.join(data_root, 'training', 'point_sem_label', '{}_sem_label.npy'.format(data_name))
            instance_mask_path = os.path.join(data_root, 'training', 'point_ins_mask', '{}_ins_mask.npy'.format(data_name))
            point = np.load(point_path)   # (x, y, z, prob, r, g, b)  N*7
            label = np.load(label_path)
            self.scene_points_list.append(point)
            self.semantic_labels_list.append(label)
            instance_mask = np.load(instance_mask_path)
            tmp, _ = np.histogram(label, range(3))
            labelweights += tmp
            instance_indices = []
            for im in range(instance_mask.shape[1]):
                ins_mask = instance_mask[:, im]
                ins_indices = np.where(ins_mask)[0]
                if len(ins_indices) > 0:
                    instance_indices.append(ins_indices)
            self.instance_indices_list.append(instance_indices)

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        scene_points = self.scene_points_list[index]
        labels = self.semantic_labels_list[index]
        instance_indices = self.instance_indices_list[index]

        collect_points_list = []
        collect_indices_list = []
        for ins_indices in instance_indices:
            if ins_indices.size > self.block_points:    # 单个实例点云数量大于block的点云数量，则需要进行进一步切分
                num_block = np.ceil(ins_indices.size/self.block_points)
                ins_points = scene_points[ins_indices]
                # 对ins_points根据z进行排序，对应对ins_indices的顺序也改变
                ins_points, ins_indices = self.sort_points_indices(ins_points, ins_indices)
                for block_id in range(int(num_block)):
                    if block_id == num_block-1: # 最后一个block要判断是否需要进行point expand
                        blk_points = ins_points[block_id*self.block_points:, :]
                        blk_indices = ins_indices[block_id*self.block_points:]
                        if blk_indices.size < self.block_points:
                            blk_points, blk_indices = self.expand_instance_point(blk_points, blk_indices)
                    else:   # 其他情况，直接获取对应的顺序的points和indices
                        blk_points = ins_points[block_id*self.block_points:(block_id+1)*self.block_points, :]
                        blk_indices = ins_indices[block_id*self.block_points:(block_id+1)*self.block_points]
                    collect_points_list.append(blk_points)
                    collect_indices_list.append(blk_indices)

            else:   # 单个实例点云数量小于block的点云数量，则需要点云数量的扩展
                ins_points = scene_points[ins_indices]
                ins_points, ins_indices = self.expand_instance_point(ins_points, ins_indices)
                collect_points_list.append(ins_points)
                collect_indices_list.append(ins_indices)
        return collect_points_list, collect_indices_list, labels

    def sort_points_indices(self, points, indices):
        """
        对点云进行z坐标的排序，对应indices也相应排序
        Args:
            points:
            indices:

        Returns:

        """
        sort_order = np.argsort(-points[:, 2])
        points_sorted = points[sort_order]
        indices_sorted = indices[sort_order]
        return points_sorted, indices_sorted

    def expand_instance_point(self, points, indices):
        """
        对数量不足对点云进行扩展
        Args:
            points:
            indices:

        Returns:

        """
        N_point = points.shape[0]
        sample_num = math.floor(N_point * self.sample_rate)
        sample_idx = np.random.choice(range(N_point), sample_num)
        sample_point = points[sample_idx, :]
        expand_num = math.ceil((self.block_points - N_point) / sample_num)

        # 将各个维度均重复
        expand_point = np.tile(sample_point, (expand_num, 1))
        expand_point = np.reshape(expand_point, (-1, 7))
        expand_point[:, :3] = expand_point[:, :3] + np.random.rand(expand_point.shape[0], 3) * 0.1
        points_exp = np.concatenate((points, expand_point[:(self.block_points-points.shape[0])]), axis=0)
        expand_indices = -np.ones((self.block_points-points.shape[0],))
        indices_exp = np.concatenate((indices, expand_indices))
        return points_exp, indices_exp

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    # KittiInstanceDatasetWholeScene
    # '''
    data_root = '/data/szy4017/code/Pointnet_Pointnet2_pytorch/data/kitti_instance'
    TEST_DATASET_WHOLE_SCENE = KittiInstanceDatasetWholeScene(data_root, block_points=1024, split='test', sample_rate=0.3)

    num_scenes = len(TEST_DATASET_WHOLE_SCENE)
    for scene_id in range(num_scenes):
        print('scene id:', scene_id)
        collect_points_list, collect_indices_list, labels = TEST_DATASET_WHOLE_SCENE.__getitem__(scene_id)
        print('labels shape:', labels.shape)
        for points, indices in zip(collect_points_list, collect_indices_list):
            print('points data shape:', points.shape)
            print('indices data shape:', indices.shape)


    # '''
