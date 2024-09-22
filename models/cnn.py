import torchvision.models as models
import torch.nn as nn
import torch
from torchsummary import summary
# feel free to add to the list: https://pytorch.org/docs/stable/torchvision/models.html
from datetime import datetime
import timm
### TWO HEAD MODELS ###
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from torch import nn






#### Identity Layer ####
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



class vit_base_patch16_224(nn.Module):
    def __init__(self, hparams):
        super(vit_base_patch16_224, self).__init__()
        # self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)#(pretrained=hparams.pretrained)
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        # Step 1: Load the model object
        TIMM = torch.load(r'C:\Users\shinp\Desktop\work\surg_vu\Trained_VIT_Autolaparo.pth')
        # TIMM = torch.load(r'C:\Users\shinp\Desktop\work\surg_vu\Trained_VIT_Cholec80.pth')
        # Step 2: Extract the state dictionary
        state_dict = TIMM.state_dict()
        # Step 3: Remove the 'model.' prefix from the state_dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '')  # Remove the 'model.' prefix
            new_state_dict[new_key] = v
        self.model.load_state_dict(new_state_dict)
        # replace final layer with number of labels
        self.model.fc = Identity()
        self.fc_phase = nn.Linear(768, hparams.out_features)
     

    def forward(self, x):
        now = datetime.now()
        out_stem = self.model(x)
       
        step = self.fc_phase(out_stem)
 
        return out_stem, step

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        resnet50model_specific_args = parser.add_argument_group(
            title='resnet50model specific args options')
        resnet50model_specific_args.add_argument("--pretrained",
                                                 action="store_true",
                                                 help="pretrained on imagenet")
        resnet50model_specific_args.add_argument(
            "--model_specific_batch_size_max", type=int, default=80)
        return parser
    