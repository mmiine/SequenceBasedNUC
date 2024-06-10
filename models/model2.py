import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x

class LSTM_Processor(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers):
        super(LSTM_Processor, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # Process the entire sequence and return only the output of the last sequence element
        output, (h_n, c_n) = self.lstm(x)
        # Take the output of the last sequence element
        last_output = output[:, -1, :]
        last_output = self.fc(last_output)
        return last_output
    

class ImageDecoder(nn.Module):
    def __init__(self, input_size, output_channels ):
        super(ImageDecoder, self).__init__()

        self.linear = nn.Sequenial(
            nn.Linear(input_size, 2048),
            nn.Linear(2048, 5000)
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, output_channels, 3, 2, 1, 1),
        )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 8, 25, 25)
        x = self.upsample(x)
        x = self.pool(x)
        return x

class SequenceNUCNet2(nn.Module):
    def __init__(self, img_size):
        super(SequenceNUCNet2, self).__init__()
        self.encoder = CNN_Encoder()
        feature_size = 12 * 12 * 64
        self.decoder = LSTM_Processor(num_features=feature_size, hidden_size=1024, num_layers=2)
        self.decoder_A = ImageDecoder(input_size=1024, output_channels = 1)
        self.decoder_B = ImageDecoder(input_size=1024, output_channels = 1)

        self.img_size = img_size

    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.shape
        # Process each frame through the CNN
        c_out = [self.encoder(x[:, i]) for i in range(seq_len)]
        c_out = torch.stack(c_out, dim=1)
        c_out = c_out.reshape(batch_size, seq_len, -1)  # Flatten spatial dimensions for LSTM
        
        # LSTM processing, taking only the last output for decoding
        lstm_out = self.decoder(c_out)
        
        # Decode outputs to A and B
        A = self.decoder_A(lstm_out[:, -1:, :])
        B = self.decoder_B(lstm_out[:, -1:, :])
        
        # Reshape to match image dimensions (assuming labels A and B are the same size as input images)
        A = A.view(batch_size, *self.img_size)
        B = B.view(batch_size, *self.img_size)
        
        return A, B