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

class LSTM_Decoder(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers):
        super(LSTM_Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_features)  # Adjust output size to match feature size

    def forward(self, x):
        # Process the entire sequence and return only the output of the last sequence element
        output, (h_n, c_n) = self.lstm(x)
        # Take the output of the last sequence element
        last_output = output[:, -1, :]
        last_output = self.fc(last_output)
        return last_output

class SequenceNUCNet1(nn.Module):
    def __init__(self, img_size):
        super(SequenceNUCNet1, self).__init__()
        self.encoder = CNN_Encoder()
        # Assuming img_size is reduced to 8x8x64 after encoding (you need to calculate based on your CNN structure)
        feature_size = 8 * 8 * 64
        self.decoder = LSTM_Decoder(num_features=feature_size, hidden_size=512, num_layers=2)
        self.fc_A = nn.Linear(feature_size, img_size[0] * img_size[1])  # Output layer for A
        self.fc_B = nn.Linear(feature_size, img_size[0] * img_size[1])  # Output layer for B

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
        A = self.fc_A(lstm_out)
        B = self.fc_B(lstm_out)
        
        # Reshape to match image dimensions (assuming labels A and B are the same size as input images)
        A = A.view(batch_size, *self.img_size)
        B = B.view(batch_size, *self.img_size)
        
        return A, B