import torch
import torch.nn as nn
import numpy as np

def train(model, criterion, optimizer, train_dataloader, num_epoch, device):
    model.to(device)
    avg_train_loss, avg_train_acc = [], []
    
    for epoch in range(num_epoch):
        model.train()
        batch_train_loss, batch_train_acc = train_one_epoch(model, criterion, optimizer, train_dataloader, device)
        avg_train_acc.append(np.mean(batch_train_acc))
        avg_train_loss.append(np.mean(batch_train_loss))

        print(f'\nEpoch [{epoch}] Average training loss: {avg_train_loss[-1]:.4f}, '
              f'Average training accuracy: {avg_train_acc[-1]:.4f}')

    return model


def train_one_epoch(model, criterion, optimizer, train_dataloader, device):
    batch_train_loss, batch_train_acc = [], []

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Zero gradients, backpropagation, and optimization
        optimizer.zero_grad()  # Better to clear gradients before loss.backward()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, pred = torch.max(outputs, 1)
        _, labels_max = torch.max(labels, 1)  # Using torch.argmax(labels, 1) for one-hot labels
        acc = (pred == labels_max).float().mean().item()

        # Store loss and accuracy for this batch
        batch_train_loss.append(loss.item())
        batch_train_acc.append(acc)
    
    return batch_train_loss, batch_train_acc


def test(model, test_dataloader, device):
    model.to(device)
    model.eval()
    batch_test_acc = []
    
    with torch.no_grad():  # Disable gradient computation during testing
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Accuracy calculation
            _, pred = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)  # Using torch.argmax(labels, 1) for one-hot labels
            acc = (pred == labels_max).float().mean().item()

            # Store accuracy for this batch
            batch_test_acc.append(acc)

    # Print average test accuracy
    print(f"The test accuracy is {torch.mean(torch.tensor(batch_test_acc)):.4f}.\n")