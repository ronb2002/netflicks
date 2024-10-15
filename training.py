import torch

import consts
from dataset_creation import get_clips_and_labels
import numpy as np
from torch.utils.data import DataLoader

from model import clip_label_list, EfficientNetLSTMModel, WeightedBCELoss

input_size = 1280
output_size = 1
batch_size = 8


def data_splits():
    clips, labels = get_clips_and_labels()
    indices = np.random.permutation(clips.shape[0])
    clips = clips[indices]
    labels = labels[indices]
    eval_clips = clips[consts.TRAIN_TEST_SPLIT:]
    eval_labels = labels[consts.TRAIN_TEST_SPLIT:]
    clips = clips[:consts.TRAIN_TEST_SPLIT]
    labels = labels[:consts.TRAIN_TEST_SPLIT]
    return clips, labels, eval_clips, eval_labels


def train(model, data):
    model.train()  # Set the model to training mode

    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    cnt = 0
    for batch in train_loader:
        b_clips, b_labels = batch
        b_clips = b_clips.to(consts.DEVICE)
        b_labels = b_labels.to(consts.DEVICE)
        b_labels = b_labels.unsqueeze(-1).float()
        optimizer.zero_grad()  # Clear previous gradients

        # Step 1: Make predictions using the model
        output = model(b_clips)  # Calls the forward function
        output = torch.sigmoid(output)
        cnt += batch_size
        if cnt % (20 * batch_size) == 0:
            print("Finished " + str(cnt) + " clips out of " + str(len(clips)))
        # Step 2: Compute the loss
        loss = loss_fn(output, b_labels)

        # Step 3: Backward pass to compute gradients
        loss.backward()

        # Step 4: Optimization step (update the weights of both EfficientNet and LSTM)
        optimizer.step()

        print(f'Loss: {loss.item(): .4f}')

        return model


def evaluate(model, data, labeled=True):
    eval_loader = DataLoader(data, batch_size=len(data), shuffle=False)
    model.eval()
    if labeled:
        b_clips, b_labels = next(iter(eval_loader))
        b_clips = b_clips.to(consts.DEVICE)
        b_labels = b_labels.to(consts.DEVICE)
        b_labels = b_labels.unsqueeze(-1).float()
    else:
        b_clips = next(iter(eval_loader))
        b_clips = b_clips.to(consts.DEVICE)
    with torch.no_grad():
        output = model(b_clips)
        output_loss = torch.sigmoid(output)
        predictions = (torch.sigmoid(output) > 0.5).long()
    if labeled:
        loss_e = loss_eval(output_loss, b_labels)
        stats(b_labels, predictions)
        print(f'Eval Loss: {loss_e.item():.4f}')
    return predictions


def stats(b_labels, predictions):
    tp, fn, fp, tn = 0, 0, 0, 0
    has_goal_pred_goal, has_goal_pred_no_goal, has_no_goal_pred_goal, has_no_goal_pred_no_goal = 0, 0, 0, 0
    for i in range(len(b_labels)):
        if 1 in b_labels[i]:
            if 1 in predictions[i]:
                has_goal_pred_goal += 1
            else:
                has_goal_pred_no_goal += 1
        else:
            if 1 in predictions[i]:
                has_no_goal_pred_goal += 1
            else:
                has_no_goal_pred_no_goal += 1
    for i in range(len(b_labels)):
        for j in range(len(b_labels[0])):
            if b_labels[i][j].item() == 1:
                if predictions[i][j].item() == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if predictions[i][j].item() == 1 and 1 not in b_labels[i,
                                                              max(0, j - 5):min(j + 5, len(b_labels[0]))]:
                    fp += 1
                else:
                    tn += 1
    print(f"True Positives: {tp}, False Negatives: {fn}, False Positives: {fp}, True Negatives: {tn}")
    print(has_goal_pred_goal, has_goal_pred_no_goal, has_no_goal_pred_goal, has_no_goal_pred_no_goal)
    if fn < consts.BEST_FN and fp < consts.BEST_FP:
        print("saving model!")
        torch.save(model, consts.FILE_PATH + f"model_{fn}_{fp}.pth")


if __name__ == "__main__":
    clips, labels, eval_clips, eval_labels = data_splits()

    print(
        f"Trying with hidden_size={consts.HIDDEN_SIZE}, num_layers={consts.NUM_LAYERS}, lr={consts.LR}, weight_fn={consts.WEIGHT_FN}, num_epochs={consts.NUM_EPOCHS}")

    labeled_clips = clip_label_list(clips, labels)
    labeled_eval_clips = clip_label_list(eval_clips, eval_labels)
    model = EfficientNetLSTMModel(input_size, consts.HIDDEN_SIZE, consts.NUM_LAYERS, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=consts.LR, weight_decay=0.0001)

    loss_fn = WeightedBCELoss(consts.WEIGHT_FN, weight_fp=1.0)
    loss_eval = WeightedBCELoss(consts.WEIGHT_FN, weight_fp=1.0)

    model = model.to(consts.DEVICE)

    # Freeze layers 0 to 3 of efficientNet, we don't train them:

    for name, param in model.efficientnet.named_parameters():
        if "features" in name and int(name.split('.')[1]) <= 3:
            param.requires_grad = False

    # Training loop
    for epoch in range(consts.NUM_EPOCHS):
        print("starting epoch " + str(epoch + 1))
        model = train(model, labeled_clips)
        evaluate(model, labeled_eval_clips)
