from os.path import expanduser

HOME = expanduser('~')

BACKBONE = 'ResNet50'
SAVE_EVERY = 5
TEST_EVERY = 1
MAX_CHECKPOINTS = 200
END_EPOCH = 200
INIT_LR = 0.001
LR_MILESTONES = [50, 100, 150]
LR_DECAY_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
STRIDE = 32
FEATURE_DIM = 512
IMAGE_SIZE = 224
ckpt_path = ''

# Path to the global-view extractor model, which also serves as a pretrained backbone for the disjoint encoder.
# Adopted from MMAL --
# Paper: https://arxiv.org/pdf/2003.09150.pdf
# Code: https://github.com/ZF4444/MMAL-Net
PRETRAINED_EXTRACTOR_PATH = "/root/autodl-tmp/fine/my_method/resnet50-19c8e357.pth"
