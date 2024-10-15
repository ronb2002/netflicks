import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as tf


def pad_and_crop(frame, target_size=(224, 224)):
    # Padding code (unchanged)
    _, _, h, w = frame.shape
    diff = abs(h - w)

    if h > w:
        padding = (diff // 2, 0, diff - (diff // 2), 0)
    else:
        padding = (0, diff // 2, 0, diff - (diff // 2))

    padded_frame = tf.pad(frame, padding)
    cropped_frame = tf.center_crop(padded_frame, target_size)

    # Convert to float and normalize (Expected for EfficientNet)
    cropped_frame = cropped_frame.float()  # Convert to float32
    cropped_frame = cropped_frame / 255.0  # Normalize pixel values to [0, 1]

    mean = [0.485, 0.456, 0.406]  # EfficientNet ImageNet mean
    std = [0.229, 0.224, 0.225]  # EfficientNet ImageNet std
    transform = tf.normalize(cropped_frame, mean, std)

    return transform


class EfficientNetLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EfficientNetLSTMModel, self).__init__()

        # Load the pretrained EfficientNet model
        self.efficientnet = models.efficientnet_b0(pretrained=True)

        # Remove the final classification layer from EfficientNet
        self.efficientnet.classifier = nn.Identity()

        # Define the LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Fully connected layer for final classification (goal/no goal)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional LSTM

    def forward(self, x):
        # Input x: (batch_size, sequence_length, height, width, channels)
        batch_size, sequence_length, _, _, _ = x.shape

        # Permute to match EfficientNet input format: (batch_size, sequence_length, channels, height, width)
        x = x.permute(0, 1, 4, 2, 3)

        features = []
        for i in range(sequence_length):
            frame = x[:, i, :, :, :]  # Extract each frame (batch_size, channels, height, width)

            # Pad and resize to 224x224
            frame = pad_and_crop(frame, target_size=(224, 224))

            # Pass frame through EfficientNet
            frame_features = self.efficientnet(frame)
            features.append(frame_features)

        # Stack features to create a sequence tensor: (batch_size, sequence_length, feature_size)
        features = torch.stack(features, dim=1)

        # Pass the sequence of features through the LSTM
        lstm_out, _ = self.lstm(features)

        # Use the output of the last time step for classification
        final_output = self.fc(lstm_out)

        return final_output


def clip_label_list(clips, labels):
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
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        # Apply weights: FN gets more weight, FP gets less
        weights = targets * self.weight_fn + (1 - targets) * self.weight_fp
        return (weights * bce_loss).mean()
