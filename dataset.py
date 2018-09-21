import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from skimage.io import imread


class ImageFolderDataset(Dataset):
    def __init__(self, root, cache=None, size=(720, 1280)):
        self.size = size
        if cache is not None and os.path.isfile(cache):
            with open(cache, 'rb') as f:
                self.root, self.images = pickle.load(f)
        else:
            self.images = sorted(os.listdir(os.path.join(root, 'test_real')))
            self.root = root
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump((self.root, self.images), f)
        
        np.savetxt('linux_database.txt', self.images, fmt='%s')

    # TODO: rewrite this part
    def __getitem__(self, item):
        name = self.images[item]
        real_img = imread(os.path.join(self.root, 'test_real', name))
        fake_img = imread(os.path.join(self.root, 'test_sync', name))
        return real_img, fake_img

    def __len__(self):
        return len(self.images)


class FaceCropDataset(Dataset):
    def __init__(self, image_dataset, pose_file, transform, crop_size=96):
        self.image_dataset = image_dataset
        self.transform = transform
        self.crop_size = crop_size

    # now pose is stored in a full .npy file!
        if not os.path.isfile(pose_file):
            raise(FileNotFoundError('Cannot find pose data...'))
        self.poses = np.load(pose_file)

    def get_full_sample(self, item):
        # skip over bad items
        while True:
            real_img, fake_img = self.image_dataset[item]
            pose = self.poses[item, ...]
            head_pos = pose[0, :]
            if head_pos[0] == -1 or head_pos[1] == -1:
                item = (item + 1) % len(self.image_dataset)
            else:
                break

        # crop head image (default 64*64)
        size = self.image_dataset.size
        left = int(head_pos[1] - self.crop_size / 2)  # don't suppose left will go out of bound eh?
        left = left if left >= 0 else 0
        left = size[1] - self.crop_size if left + self.crop_size > size[1] else left

        top = int(head_pos[0] - self.crop_size / 2)
        top = top if top >= 0 else 0
        top = size[0] - self.crop_size if top + self.crop_size > size[0] else top

        real_head = real_img[top: top + self.crop_size, left: left + self.crop_size, :]
        fake_head = fake_img[top: top + self.crop_size, left: left + self.crop_size, :]
        #
        # from matplotlib.pyplot import imshow, show
        # imshow(real_head)
        # show()
        # imshow(fake_head)
        # show()

        # keep full fake image to visualize enhancement result
        return self.transform(real_head), self.transform(fake_head), \
               top, top + self.crop_size, \
               left, left + self.crop_size, \
               real_img, fake_img

    def __getitem__(self, item):
        real_head, fake_head, _, _, _, _, _, _ = self.get_full_sample(item)
        return {'real_heads': real_head, 'fake_heads': fake_head}

    def __len__(self):
        return len(self.image_dataset)

