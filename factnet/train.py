import os.path as osp
import factnet.archs
import factnet.data
import factnet.models
import factnet.losses

from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
