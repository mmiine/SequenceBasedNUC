import torch
import torch.nn as nn
import torch.nn.functional as F
from models.convlstm import ConvLSTM


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        return enc1, enc2, enc3, enc4, self.pool4(enc4)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(768, 256)  # 512 (from enc4) + 256 (upsampled x)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(384, 128)  # 256 (from enc3) + 128 (upsampled x)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(192, 64)   # 128 (from enc2) + 64 (upsampled x)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(96, 64)    # 64 (from enc1) + 32 (upsampled x)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, enc1, enc2, enc3, enc4):
        x = self.upconv1(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.decoder1(x)

        x = self.upconv2(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder2(x)

        x = self.upconv3(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder3(x)

        x = self.upconv4(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder4(x)

        x = self.final(x)
        return x



class UNetConvLSTM(nn.Module):
    def __init__(self, image_size):
        super(UNetConvLSTM, self).__init__()
        self.encoder = Encoder()
        self.convlstm = ConvLSTM(input_dim=512, hidden_dim=512, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.decoderA = Decoder()
        self.decoderB = Decoder()


    def forward(self, x):
        batch_size, seq_len, height, width = x.size()
        x = x.view(batch_size * seq_len, 1, height, width)
        
        # Encoder
        enc1, enc2, enc3, enc4, x = self.encoder(x)
        x = x.view(batch_size, seq_len, 512, height // 16, width // 16)
        
        # ConvLSTM
        layer_output_list, last_state_list = self.convlstm(x)
        x = layer_output_list[-1][:, -1, :, :, :]  # Output of the last layer and last time step
        
        # Decoder
        enc1 = enc1.view(batch_size, seq_len, 64, height, width)[:, -1]
        enc2 = enc2.view(batch_size, seq_len, 128, height // 2, width // 2)[:, -1]
        enc3 = enc3.view(batch_size, seq_len, 256, height // 4, width // 4)[:, -1]
        enc4 = enc4.view(batch_size, seq_len, 512, height // 8, width // 8)[:, -1]
        outputA = self.decoderA(x, enc1, enc2, enc3, enc4)
        outputB = self.decoderB(x, enc1, enc2, enc3, enc4)
        return outputA, outputB