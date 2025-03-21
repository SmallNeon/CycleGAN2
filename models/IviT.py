import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
# from .conv_transformer import Transformer,Encoder,Decoder
import functools

class Encoder(nn.Module):
    def __init__(self, input_nc, ngf=16, norm_layer=nn.BatchNorm2d, n_downsampling=3):
        super(Encoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),

                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        self.down_sampling = nn.Sequential(*model)

    def forward(self, input):
        input = self.down_sampling(input)
        return input


class Decoder(nn.Module):
    def __init__(self, output_nc, ngf=16, norm_layer=nn.BatchNorm2d, n_downsampling=3):
        super(Decoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        self.up_sampling = nn.Sequential(*model)

    def forward(self, input):
        input = self.up_sampling(input)
        return input


"""Basic block with convolution, normalization, and activation"""
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
                        nn.InstanceNorm2d(out_channels),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.LeakyReLU(inplace=True),
                        nn.InstanceNorm2d(out_channels),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.LeakyReLU(inplace=True)
                        )
                        
    def forward(self, x):
        return self.block(x)
        
"""Down-sampling block"""
class DownsampleBlock(nn.Module):
    def __init__(self,features):
        super(DownsampleBlock, self).__init__()
        self.downsample = nn.Conv2d(features, features, kernel_size=3,stride=2,padding=1)

    def forward(self,x):
        return self.downsample(x)
    
"""Up-sampling block"""
class UpsampleBlock(nn.Module):
    def __init__(self,features):
        super(UpsampleBlock, self).__init__()
        self.upconv = nn.Sequential(
        nn.Upsample(scale_factor = 2),
        nn.Conv2d(features, features, kernel_size = 3, padding = 1),
    )
    def forward(self,x):
        return self.upconv(x)

class FourierEmbedding(nn.Module):
    # arXiv: 2011.13775

    def __init__(self, embed_dim):
        super(FourierEmbedding, self).__init__()
        self.projector = nn.Linear(2, embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape   
        x = torch.arange(w).to(torch.float32)
        y = torch.arange(h).to(torch.float32)
        x, y = torch.meshgrid(x, y)
        x = x.reshape((1, -1))
        y = y.reshape((1, -1))

        x_norm = 2 * x / (w  - 1) - 1
        y_norm = 2 * y / (h - 1) - 1

        z = torch.cat((x_norm.unsqueeze(2), y_norm.unsqueeze(2)), dim = 2)
        z = z.to(device='cuda')

        return torch.sin(self.projector(z))
    

"Feedforward block with specified convolutions and operations"
class DepthWiseFFN(nn.Module):
    def __init__(self, in_dim,expand_ratio=4,stride=1):
        super(DepthWiseFFN, self).__init__()
        kernel_size = 3
        hidden_dim  = in_dim*expand_ratio
        layers = []
        '1-d conv layer'
        layers.extend([
        nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU6(inplace=True)])

        'depth-wise conv layer'
        dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ]
        layers.extend(dp)

        "1-d conv layer"
        layers.extend([
        nn.Conv2d(hidden_dim, in_dim, 1, 1, 0, bias=False),
        nn.BatchNorm2d(in_dim)])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        residual = x  
        x = self.conv(x)
        return x + residual   

class PositionWiseFFN(nn.Module):
    def __init__(self, features,expand_ratio=4):
        super(PositionWiseFFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(features, 1536),
            nn.GELU(),
            nn.Linear(1536, features),
        )

    def forward(self, x):
        return self.net(x)

"""Transformer encoder block"""
class TransformerEncoderBlock(nn.Module):
    def __init__(self, features,num_heads,rezero):
        super(TransformerEncoderBlock, self).__init__()
        self.layernorm1 = nn.LayerNorm(features)
        self.atten = nn.MultiheadAttention(features, num_heads, batch_first=True)
        self.layernorm2 = nn.LayerNorm(features)
        self.ffn = DepthWiseFFN(features)
        self.rezero = rezero
        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self,x):
        batch_size, num_token, embed_dim  = x.shape                                 
        patch_size = int(math.sqrt(num_token))

        y1 = self.layernorm1(x)
        y1, _ = self.atten(x, x, x)
        y = self.re_alpha*y1 + x

        y2 = self.layernorm2(y) # (b,h*w,dim)
        y2 = y2.transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size) # (b, dim, h, w)
        y2 = self.ffn(y2).flatten(2).transpose(1, 2) # (b,h*w,dim)
        # y2 = self.ffn(y2)

        y = self.re_alpha*y2 + y 
        return y
    
"""Vision Transformer block"""
class ViT(nn.Module):

    def __init__(self, embed_dim, input_features, features,num_layers, num_heads,rezero):
        super(ViT, self).__init__()
        # self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.positional_embedding = FourierEmbedding(embed_dim)
        self.encoder = nn.Sequential(*[TransformerEncoderBlock(features,num_heads,rezero) for _ in range(num_layers)])
        self.layernorm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim + input_features, features)
        self.trans_output = nn.Linear(features, input_features)

    def forward(self, x):
        batch_size, embed_features, patch_size, patch_size  = x.shape                                 
    
        "positional encoding"
        pos = self.positional_embedding(x) # (b, embed_features, h, w)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (b, h*w, embed_features)
        pos = pos.expand((x.shape[0], *pos.shape[1:]))
        x = torch.cat([pos, x], dim = 2) # (b, h*w, embed_dim+embed_features).
        x = self.output(x) # (b, h*w, features)

        'vit encoder block'
        x = self.encoder(x) # (b, h*w, features)
        x = self.trans_output(x) # (b, h*w, embed_features)
        x = x.transpose(1, 2).view(batch_size, embed_features, patch_size, patch_size) # (b, embed_features, h, w)

        return x
    
"""IViT-CycleGAN model generator section"""
class IViT_CycleGAN(nn.Module):
    def __init__(self,in_channels=1, out_channels=1, features=32):
        super(IViT_CycleGAN, self).__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size = 3, padding = 1),
            nn.LeakyReLU(inplace=True))
        
        self.encoder1 = BasicBlock(features, features)
        self.downsample1 = DownsampleBlock(features)
        self.encoder2 = BasicBlock(features, features*2)
        self.downsample2 = DownsampleBlock(features*2)
        self.encoder3 =  BasicBlock(features*2, features*4)
        self.downsample3 = DownsampleBlock(features*4)
        self.encoder4 = BasicBlock(features * 4, features * 8)
        self.downsample4 = DownsampleBlock(features * 8)


        self.vit = ViT(embed_dim=384, input_features=256,features=384, num_layers=12, num_heads=6,rezero=True)
        # self.vit = Transformer(dim=8*16, proj_kernel=3, kv_proj_stride=2, depth=3, heads=6,
        #                                mlp_mult=4, dropout=0)

        self.upconv4 = UpsampleBlock(features * 8)
        self.decoder4 = BasicBlock(features * 16, features * 4)
        self.upconv3 = UpsampleBlock(features * 4)
        self.decoder3 = BasicBlock(features * 8, features * 2)
        self.upconv2 = UpsampleBlock(features * 2)
        self.decoder2 = BasicBlock(features * 4, features)
        self.upconv1 = UpsampleBlock(features)
        self.decoder1 = BasicBlock(features*2, features)

        self.postprocess = nn.Sequential(
            nn.Conv2d(features, out_channels, kernel_size=1),
            nn.Tanh()
        )
        self.conv_encoder = Encoder(input_nc=1, ngf=16, n_downsampling=3)
        self.conv_decoder = Decoder(output_nc=1, ngf=16, n_downsampling=3)


    def forward(self, x):
        x = self.preprocess(x) 
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.downsample1(enc1))
        enc3 = self.encoder3(self.downsample2(enc2))
        enc4 = self.encoder4(self.downsample3(enc3))
        x = self.downsample4(enc4)

        # x = self.conv_encoder(x)
        x = self.vit(x)
        # x = self.conv_decoder(x)

        dec4 = self.upconv4(x)
        dec4 = torch.cat((dec4, enc4), dim=1) 
        dec4 = self.decoder4(dec4)        
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.postprocess(dec1)
        # return x