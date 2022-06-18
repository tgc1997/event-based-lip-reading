import glob
import json
import os
import torch
from torch.utils.data import Dataset
from .cvtransforms import *

# https://github.com/uzh-rpg/rpg_e2vid/blob/d0a7c005f460f2422f2a4bf605f70820ea7a1e5f/utils/inference_utils.py#L480
def events_to_voxel_grid_pytorch(events, num_bins, width, height, device):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    with torch.no_grad():

        events_torch = torch.from_numpy(events).float()
        events_torch = events_torch.to(device)

        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device)
        if events_torch.shape[0] == 0:
            return voxel_grid

        voxel_grid = voxel_grid.flatten()

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events_torch[-1, 0]
        first_stamp = events_torch[0, 0]
        deltaT = float(last_stamp - first_stamp)

        if deltaT == 0:
            deltaT = 1.0

        events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
        ts = events_torch[:, 0]
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices]
                                    * width + tis_long[valid_indices] * width * height,
                              source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices] * width
                                    + (tis_long[valid_indices] + 1) * width * height,
                              source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid

def events_to_voxel_all(events, frame_nums, seq_len, num_bins, width, height, device):
    voxel_len = min(seq_len, frame_nums) * num_bins
    voxel_grid_all = np.zeros((num_bins * seq_len, 1, height, width))
    voxel_grid = events_to_voxel_grid_pytorch(events, voxel_len, width, height, device)
    voxel_grid = voxel_grid.unsqueeze(1).cpu().numpy()
    voxel_grid_all[:voxel_len] = voxel_grid
    return voxel_grid_all

class DVS_Lip(Dataset):
    def __init__(self, phase, args):
        self.labels = sorted(os.listdir(os.path.join(args.event_root, phase)))
        self.length = args.seq_len
        self.phase = phase
        self.args = args

        self.file_list = sorted(glob.glob(os.path.join(args.event_root, phase, '*', '*.npy')))

        with open('./data/frame_nums.json', 'r') as f:
            self.frame_nums = json.load(f)

    def __getitem__(self, index):
        # load timestamps
        word = self.file_list[index].split('/')[-2]
        person = self.file_list[index].split('/')[-1][:-4]
        frame_num = self.frame_nums[self.phase][word][int(person)]

        # load events
        try:
            events_input = np.load(self.file_list[index])
        except:
            print(self.file_list[index])
        events_input = events_input[np.where((events_input['x'] >= 16) & (events_input['x'] < 112) & (events_input['y'] >= 16) & (events_input['y'] < 112))]
        events_input['x'] -= 16
        events_input['y'] -= 16
        t, x, y, p = events_input['t'], events_input['x'], events_input['y'], events_input['p']
        events_input = np.stack([t, x, y, p], axis=-1)

        # convert events to voxel_grid
        event_voxel_low = events_to_voxel_all(events_input, frame_num, self.length, self.args.num_bins[0], 96, 96, device='cpu') # (30*num_bins[0], 96, 96)
        event_voxel_high = events_to_voxel_all(events_input, frame_num, self.length, self.args.num_bins[1], 96, 96, device='cpu') # (30*num_bins[1], 96, 96)

        # data augmentation
        if self.phase == 'train':
            event_voxel_low, event_voxel_high = RandomCrop(event_voxel_low, event_voxel_high, (88, 88))
            event_voxel_low, event_voxel_high = HorizontalFlip(event_voxel_low, event_voxel_high)
        else:
            event_voxel_low, event_voxel_high  = CenterCrop(event_voxel_low, event_voxel_high, (88, 88))

        result = {}
        result['event_low'] = torch.FloatTensor(event_voxel_low)
        result['event_high'] = torch.FloatTensor(event_voxel_high)
        result['label'] = self.labels.index(word)

        return result

    def __len__(self):
        return len(self.file_list)
