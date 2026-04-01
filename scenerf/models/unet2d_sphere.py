"""
Code adapted from https://github.com/cv-rits/MonoScene/blob/master/monoscene/models/unet2d.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, channel_num, dilations):
        super(BasicBlock, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3,
                      padding=dilations[0], dilation=dilations[0]),
            nn.BatchNorm2d(channel_num),
            nn.LeakyReLU(),
            # nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3,
                      padding=dilations[1], dilation=dilations[1]),
            nn.BatchNorm2d(channel_num),
        )
        self.lrelu = nn.LeakyReLU()
        # self.lrelu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.lrelu(x)
        return out


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features,
                      kernel_size=3, stride=1, padding=1),
            BasicBlock(output_features, dilations=[1, 1]),
            BasicBlock(output_features, dilations=[2, 2]),
            BasicBlock(output_features, dilations=[3, 3]),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=(concat_with.shape[2], concat_with.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderSphere(nn.Module):
    def __init__(
        self,
        num_features,
        bottleneck_features,
        out_feature,
        out_img_W,
        out_img_H
    ):
        super(DecoderSphere, self).__init__()

        self.out_img_W = out_img_W
        self.out_img_H = out_img_H

        features = int(num_features)

        self.conv2 = nn.Conv2d(
            bottleneck_features, features, kernel_size=1, stride=1, padding=1
        )

        self.out_feature_1_1 = out_feature
        self.out_feature_1_2 = out_feature
        self.out_feature_1_4 = out_feature
        self.out_feature_1_8 = out_feature
        self.out_feature_1_16 = out_feature
        self.feature_1_16 = features // 2
        self.feature_1_8 = features // 4
        self.feature_1_4 = features // 8
        self.feature_1_2 = features // 16
        self.feature_1_1 = features // 32

        self.resize_1_1 = nn.Conv2d(
            3, self.feature_1_1, kernel_size=1
        )
        self.resize_1_2 = nn.Conv2d(
            32, self.feature_1_2, kernel_size=1
        )
        self.resize_1_4 = nn.Conv2d(
            48, self.feature_1_4, kernel_size=1
        )
        self.resize_1_8 = nn.Conv2d(
            80, self.feature_1_8, kernel_size=1
        )
        self.resize_1_16 = nn.Conv2d(
            224, self.feature_1_16, kernel_size=1
        )

        self.resize_output_1_1 = nn.Conv2d(
            self.feature_1_1, self.out_feature_1_1, kernel_size=1
        )
        self.resize_output_1_2 = nn.Conv2d(
            self.feature_1_2, self.out_feature_1_2, kernel_size=1
        )
        self.resize_output_1_4 = nn.Conv2d(
            self.feature_1_4, self.out_feature_1_4, kernel_size=1
        )
        self.resize_output_1_8 = nn.Conv2d(
            self.feature_1_8, self.out_feature_1_8, kernel_size=1
        )
        self.resize_output_1_16 = nn.Conv2d(
            self.feature_1_16, self.out_feature_1_16, kernel_size=1
        )

        self.up16 = UpSampleBN(
            skip_input=features + 224, output_features=self.feature_1_16
        )
        self.up8 = UpSampleBN(
            skip_input=self.feature_1_16 + 80, output_features=self.feature_1_8
        )
        self.up4 = UpSampleBN(
            skip_input=self.feature_1_8 + 48, output_features=self.feature_1_4,
        )
        self.up2 = UpSampleBN(
            skip_input=self.feature_1_4 + 32, output_features=self.feature_1_2
        )
        self.up1 = UpSampleBN(
            skip_input=self.feature_1_2 + 3, output_features=self.feature_1_1
        )

    def get_sphere_feature(self, x, pix, pix_sphere, scale):
        out_W, out_H = round(self.out_img_W/scale), round(self.out_img_H/scale)
        map_sphere = torch.zeros((out_W, out_H, 2)).type_as(x) - 10.0
        pix_sphere_scale = torch.round(pix_sphere / scale).long()
        pix_scale = pix // scale
        pix_sphere_scale[:, 0] = pix_sphere_scale[:, 0].clamp(0, out_W-1)
        pix_sphere_scale[:, 1] = pix_sphere_scale[:, 1].clamp(0, out_H-1)
  
        map_sphere[pix_sphere_scale[:, 0],
                   pix_sphere_scale[:, 1], :] = pix_scale
        map_sphere = map_sphere.reshape(-1, 2)

  
        map_sphere[:, 0] /= x.shape[3]
        map_sphere[:, 1] /= x.shape[2]
        map_sphere = map_sphere * 2 - 1
        map_sphere = map_sphere.reshape(1, 1, -1, 2)

        feats = F.grid_sample(
            x,
            map_sphere,
            align_corners=False,
            mode='bilinear'
        )
        feats = feats.reshape(feats.shape[0], feats.shape[1], out_W, out_H)
        feats = feats.permute(0, 1, 3, 2)

        return feats

    def forward(self, features, pix, pix_sphere):
        x_block1, x_block2, x_block4, x_block8, x_block16, x_block32 = (
            features[0],
            features[4],
            features[5],
            features[6],
            features[8],
            features[11],
        )
        bs = x_block32.shape[0]
        x_block32 = self.conv2(x_block32)
  
  
        x_sphere_32 = self.get_sphere_feature(x_block32, pix, pix_sphere, 32)
        
        x_sphere_16 = self.get_sphere_feature(x_block16, pix, pix_sphere, 16)
        
        x_sphere_8 = self.get_sphere_feature(x_block8, pix, pix_sphere, 8)
       
        x_sphere_4 = self.get_sphere_feature(x_block4, pix, pix_sphere, 4)
       
        x_sphere_2 = self.get_sphere_feature(x_block2, pix, pix_sphere, 2)
       
        x_sphere_1 = self.get_sphere_feature(x_block1, pix, pix_sphere, 1)
       
        x_1_16 = self.up16(x_sphere_32, x_sphere_16)
        x_1_8 = self.up8(x_1_16, x_sphere_8)
        x_1_4 = self.up4(x_1_8, x_sphere_4)
        x_1_2 = self.up2(x_1_4, x_sphere_2)
        x_1_1 = self.up1(x_1_2, x_sphere_1)

     

        return {
            "1_1": x_1_1,
            "1_2": x_1_2,
            "1_4": x_1_4,
            "1_8": x_1_8,
            "1_16": x_1_16,
        }


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UNet2DSphere(nn.Module):
    def __init__(self, backend, num_features, out_feature, out_img_H, out_img_W):
        super(UNet2DSphere, self).__init__()
        self.encoder = Encoder(backend)
        self.out_img_H = out_img_H
        self.out_img_W = out_img_W
        self.decoder = DecoderSphere(
            out_feature=out_feature,
            bottleneck_features=num_features,
            num_features=num_features,
            out_img_W=out_img_W,
            out_img_H=out_img_H
        )

    def forward(self, x, pix, pix_sphere):
        encoded_feats = self.encoder(x)
        unet_out = self.decoder(encoded_feats, pix, pix_sphere)
        return unet_out

    def get_encoder_params(self):  
        return self.encoder.parameters()

    def get_decoder_params(self):  
        return self.decoder.parameters()

    @classmethod
    def build(cls, **kwargs):
        basemodel_name = "tf_efficientnet_b7_ns"
        num_features = 2560

        print("Loading base model ()...".format(basemodel_name), end="")
        basemodel = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
        )
        print("Done.")

        # Remove last layer
        print("Removing last two layers (global_pool & classifier).")
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print("Building Encoder-Decoder model..", end="")
        m = cls(basemodel, num_features=num_features, **kwargs)
        print("Done.")
        return m