from torch import nn
import torch
from networks import resnet
from utils import constants
from utils import view_extractor as ve
import clip


class DisjointEncoder(nn.Module):
    def __init__(self, num_classes, num_local, crop_mode='five_crop'):
        super(DisjointEncoder, self).__init__()
        self.num_classes = num_classes
        self.num_local = num_local
        self.crop_mode = crop_mode
        self.extractor = resnet.resnet50(pretrained=True, pth_path=constants.PRETRAINED_EXTRACTOR_PATH)
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        global_view = ve.extract_global(x, self.extractor).to(x.device)
        with torch.no_grad():
            global_embed = self.model.encode_image(global_view)
        #global_fm, global_embed, _ = self.extractor(global_view.detach())

        local_views = ve.extract_local(global_view, self.num_local, crop_mode=self.crop_mode)
        #_, local_embeds, _ = self.extractor(local_views)
        with torch.no_grad():
            local_embeds = self.model.encode_image(local_views)
        local_embeds = local_embeds.reshape(len(x), self.num_local, -1)

        return global_embed, local_embeds

    def forward_with_views(self, x,crop_mode):
        if crop_mode == "random":
            num_local = 8
        elif crop_mode == "4":
            num_local =4
        elif crop_mode == "9":
            num_local =9
        elif crop_mode == "16":
            num_local =16
        elif crop_mode == "25":
            num_local =25
        #print(x.shape)
        global_view = ve.extract_global(x, self.extractor).to(x.device)
        #print(global_view.shape)
        with torch.no_grad():
            global_embed = self.model.encode_image(global_view)
        #global_fm, global_embed, _ = self.extractor(global_view.detach())

        local_views = ve.extract_local(global_view, num_local, crop_mode=crop_mode)
        with torch.no_grad():
            local_embeds = self.model.encode_image(local_views)
        #_, local_embeds, _ = self.extractor(local_views)
        local_embeds = local_embeds.reshape(len(x), num_local, -1)

        return global_embed, local_embeds, global_view, local_views
