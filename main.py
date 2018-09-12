import model
import dataset
from trainer import Trainer

import os

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.backends import cudnn
from PIL import Image

image_transforms = transforms.Compose([
        Image.fromarray,
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
    ])


def load_models(directory, batch_num):
    generator = model.GlobalGenerator()
    discriminator = model.NLayerDiscriminator(input_nc=3)
    gen_name = os.path.join(directory, '%05d_generator.pth' % batch_num)
    dis_name = os.path.join(directory, '%05d_discriminator.pth' % batch_num)

    if os.path.isfile(gen_name) and os.path.isfile(dis_name):
        gen_dict = torch.load(gen_name)
        dis_dict = torch.load(dis_name)
        generator.load_state_dict(gen_dict)
        discriminator.load_state_dict(dis_dict)
        print('Models loaded, resume training from batch %05d...' % batch_num)
    else:
        print('Cannot find saved models, start training from scratch...')
        batch_num = 0

    return generator, discriminator, batch_num


def main():
    # configs
    dataset_dir = 'data/ntu256_pairs'
    pose_dir = 'data/ntu256_cords'
    ckpt_dir = 'checkpoints'
    log_dir = 'logs'
    batch_num = 0
    batch_size = 64

    image_folder = dataset.ImageFolderDataset(dataset_dir, cache=os.path.join(dataset_dir, 'local.db'))
    face_dataset = dataset.FaceCropDataset(image_folder, pose_dir, image_transforms)
    data_loader = DataLoader(face_dataset, batch_size=batch_size,
                             drop_last=True, num_workers=0, shuffle=True)

    generator, discriminator, batch_num = load_models(ckpt_dir, batch_num)

    trainer = Trainer(ckpt_dir, log_dir, face_dataset, data_loader)
    trainer.train(generator, discriminator, batch_num)


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
