import torch
import torch.nn as nn
import numpy as np

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
            _, pred = torch.max(outputs, 1)  # Predictions (indices of the class with max logit)
            acc = (pred == labels).float().mean().item()  # Compare with labels as class indices

            # Store accuracy for this batch
            batch_test_acc.append(acc)

    # Print average test accuracy
    print(f"The test accuracy is {torch.mean(torch.tensor(batch_test_acc)):.4f}.\n")



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

