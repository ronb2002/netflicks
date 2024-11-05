import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as tf


def pad_and_crop(frame, target_size=(224, 224)):
    """Pads and crops the frame as part of preprocessing"""
    # Padding code
    _, _, h, w = frame.shape
    diff = abs(h - w)

    if h > w:
        padding = (diff // 2, 0, diff - (diff // 2), 0)
    else:
        padding = (0, diff // 2, 0, diff - (diff // 2))

    padded_frame = tf.pad(frame, padding)
    cropped_frame = tf.center_crop(padded_frame, target_size)

    # Convert to float and normalize
    cropped_frame = cropped_frame.float()
    cropped_frame = cropped_frame / 255.0

    mean = [0.485, 0.456, 0.406]  # EfficientNet ImageNet mean
    std = [0.229, 0.224, 0.225]  # EfficientNet ImageNet std
    transform = tf.normalize(cropped_frame, mean, std)

    return transform


class EfficientNetLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EfficientNetLSTMModel, self).__init__()

        # Loading the pretrained EfficientNet model
        self.efficientnet = models.efficientnet_b0(pretrained=True)

        # Removing the final classification layer from EfficientNet, since it isn't necessary
        self.efficientnet.classifier = nn.Identity()

        # Defining the LSTM used in our model
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Defining the fully connected layer for the final binary classification (goal/no goal)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiplying by 2 since it is a bidirectional LSTM

    def forward(self, x):
        # Input x: (batch_size, sequence_length, height, width, channels)
        batch_size, sequence_length, _, _, _ = x.shape

        # Changed order to match EfficientNet's input format
        x = x.permute(0, 1, 4, 2, 3)

        features = []
        for i in range(sequence_length):
            frame = x[:, i, :, :, :]  # Extracting information for each frame

            # Padding and resizing to 224x224
            frame = pad_and_crop(frame, target_size=(224, 224))

            # Passing the frame through EfficientNet
            frame_features = self.efficientnet(frame)
            features.append(frame_features)

        # Stacking the features to create a sequence tensor of: batch_size, sequence_length, feature_size
        features = torch.stack(features, dim=1)

        # Passing the sequence of features through the LSTM
        lstm_out, _ = self.lstm(features)

        # Using the output of the last step for classification
        final_output = self.fc(lstm_out)

        return final_output


def clip_label_list(clips, labels):
    """Turns clips and labels into one labeled-clips object"""
    labeled_clips = []
    for i in range(len(clips)):
        labeled_clips.append((clips[i], labels[i]))
    return labeled_clips


class WeightedBCELoss(nn.Module):
    def __init__(self, weight_fn, weight_fp):
        super(WeightedBCELoss, self).__init__()
        self.weight_fn = weight_fn
        self.weight_fp = weight_fp

    def forward(self, inputs, targets):
        # Using binary cross entropy (log) loss with different weights for fn and fp as the loss function
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        weights = targets * self.weight_fn + (1 - targets) * self.weight_fp
        return (weights * bce_loss).mean()
