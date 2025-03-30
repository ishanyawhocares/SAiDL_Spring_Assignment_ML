import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

DEVICE = torch.device("cpu")

NUM_CLASSES = 10 # CIFAR-10 has 10 distinct classes of images
BATCH_SIZE = 128 # Process images in batches of 128. Affects memory usage and training stability.
LEARNING_RATE = 0.001 # Controls how much the model weights are updated each step. A common starting point for Adam.
EPOCHS = 5 # Number of times to loop through the entire training dataset. Adjust based on how quickly the model converges.
GAMMA = 2.0 # Parameter for Focal Loss and NFL. Controls focus on hard examples. Standard value is 2.0.
BETA = 1.0 # Weighting factor for the passive loss in APL (ActiveLoss + BETA * PassiveLoss). 1.0 gives equal initial weighting.

SYMMETRIC_NOISE_RATES = [0.0, 0.2, 0.4, 0.6, 0.8] # η values. Include 0.0 (clean) as a baseline.
LOSS_FUNCTIONS_TO_TEST = ['CE', 'FL', 'NCE', 'NFL', 'APL-NCE-MAE', 'APL-NFL-MAE']

# Dictionary to store the best test accuracy achieved for each experiment configuration
# Key: tuple (loss_function_name, noise_rate), Value: best_accuracy
RESULTS = {}

# Transformations for the Training Set
transform_train = transforms.Compose([
    # Data Augmentation: Increase dataset variability to prevent overfitting
    transforms.RandomCrop(32, padding=4), # Randomly crop the image, padding first
    transforms.RandomHorizontalFlip(), # Randomly flip images horizontally
    # Convert image data to PyTorch Tensors (values 0.0 to 1.0)
    transforms.ToTensor(),
    # Normalize pixel values: (pixel - mean) / std_dev. Uses CIFAR-10 specific means/stds.
    # Helps stabilize training.
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Transformations for the Test Set
# No data augmentation here! We want to evaluate on the original, unmodified test images.
transform_test = transforms.Compose([
    transforms.ToTensor(), # Convert to Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Normalize
])

# --- Load Clean Datasets using Torchvision ---
# Download CIFAR-10 if not present, then load it.
# 'root': where to store data. 'train=True': get training set. 'download=True': download if needed. 'transform': apply defined transformations.
trainset_clean = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

# Load the test set - IMPORTANT: Keep this clean for evaluation!
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

# --- Create Data Loaders ---
# DataLoaders handle batching, shuffling, and parallel data loading.
# 'dataset': the dataset to load from. 'batch_size': defined above. 'shuffle=False': order doesn't matter for test set.
# 'num_workers': how many subprocesses to use for data loading (speeds things up).
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)



# --- Function to Inject Symmetric Noise ---
def symmetric_noisefy(dataset, noise_rate):

    # IMPORTANT: Create a deep copy to avoid modifying the original dataset!
    noisy_dataset = deepcopy(dataset)
    # Get labels as a NumPy array for easier manipulation
    targets = np.array(noisy_dataset.targets)
    num_samples = len(targets)
    # Determine how many labels to corrupt
    num_noisy = int(noise_rate * num_samples)

    # Get a random subset of indices to corrupt
    all_indices = np.arange(num_samples)
    noisy_indices = np.random.choice(all_indices, size=num_noisy, replace=False)

    print(f"Injecting symmetric noise: rate={noise_rate}, total samples={num_samples}, noisy samples={num_noisy}")

    # Iterate through the selected indices and flip labels
    for i in noisy_indices:
        og_label = targets[i]
        # Generate potential new labels (all classes EXCEPT the original one)
        potential_new_labels = list(range(NUM_CLASSES))
        potential_new_labels.remove(og_label)
        # Randomly choose one of the incorrect labels
        noisy_label = np.random.choice(potential_new_labels)
        targets[i] = noisy_label # Update the label in our numpy array

    # Update the dataset's targets with the modified numpy array
    noisy_dataset.targets = targets.tolist() # Convert back to list for the dataset object
    return noisy_dataset


# --- Generate and store all needed noisy datasets ---
noisy_trainsets = {}
for eta in SYMMETRIC_NOISE_RATES:
    print(f"Generating dataset with symmetric noise rate: {eta}")
    # Create and store each noisy dataset, keyed by its noise rate
    noisy_trainsets[eta] = symmetric_noisefy(trainset_clean, eta)

print("Finished generating noisy datasets.")


# --- Define the Custom CNN Architecture ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__() # Initialize the parent nn.Module class

        # --- Convolutional Block 1 ---
        # Input: 3 channels (RGB), Output: 32 channels (feature maps)
        # Kernel size: 3x3. Padding: 1 (preserves width/height with stride 1).
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Activation function: ReLU introduces non-linearity
        self.relu1 = nn.ReLU()
        # Pooling: Reduces spatial dimensions (32x32 -> 16x16). Kernel size 2x2, Stride 2.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 2 ---
        # Input: 32 channels (from prev block), Output: 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # Pooling: Reduces spatial dimensions (16x16 -> 8x8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 3 ---
        # Input: 64 channels, Output: 128 channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        # Pooling: Reduces spatial dimensions (8x8 -> 4x4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Flattening Layer ---
        # Flattens the 3D feature map (128 channels x 4 width x 4 height) into a 1D vector
        self.flatten = nn.Flatten()
        # Calculate the flattened size: 128 * 4 * 4 = 2048

        # --- Fully Connected (Linear) Block ---
        # Input: 2048 features (from flatten), Output: 128 features
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=128)
        self.relu4 = nn.ReLU() # Activation

        # --- Output Layer ---
        # Input: 128 features, Output: num_classes (10) features.
        # These are the raw scores (logits) for each class. Softmax will be applied in the loss function.
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # Defines the forward pass: how input `x` flows through the layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x) # Flatten before fully connected layers

        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x) # Output logits
        return x

# test_model = SimpleCNN(num_classes=NUM_CLASSES)
# print(test_model)
# del test_model 





# --- Standard Cross-Entropy Loss ---
criteria = nn.CrossEntropyLoss()

# --- Focal Loss ---
class FocalLoss(nn.Module):
    """ Implementation of Focal Loss """
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # reduction: 'mean', 'sum', or 'none' (return loss per element)
        self.reduction = reduction

    def forward(self, logits, target):
        # Calculate standard Cross Entropy loss but without reduction
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        # Calculate pt (probability of the true class)
        # pt = exp(-ce_loss), because ce_loss = -log(pt)
        pt = torch.exp(-ce_loss)
        # Calculate Focal Loss: -(1-pt)^gamma * log(pt) = (1-pt)^gamma * ce_loss
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

# --- Normalized Cross Entropy (NCE) - Simple Bounded Version ---
# Loss = 1 - probability_of_true_class
def nce_loss(logits, target, reduction='mean'):
    # Apply softmax to convert logits to probabilities
    probs = F.softmax(logits, dim=1)
    # Get the probabilities corresponding to the true labels
    # probs[rows, columns] -> select p_i[y_i] for each sample i in batch
    correct_probs = probs[torch.arange(logits.size(0)), target]
    # Calculate the loss: 1 - pt
    loss = 1.0 - correct_probs
    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# --- Normalized Focal Loss (NFL) - Simple Bounded Version ---
# Loss = (1 - probability_of_true_class)^gamma
def nfl_loss(logits, target, gamma=2.0, reduction='mean'):
    probs = F.softmax(logits, dim=1)
    correct_probs = probs[torch.arange(logits.size(0)), target]
    # Calculate the loss: (1 - pt)^gamma
    loss = (1.0 - correct_probs) ** gamma
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# --- Mean Absolute Error (MAE) Loss - Passive Component for APL ---
def mae_loss_passive(logits, target, num_classes=10, reduction='mean'):
    probs = F.softmax(logits, dim=1)
    # Convert target labels (e.g., 3) to one-hot vectors (e.g., [0,0,0,1,0,0,0,0,0,0])
    target_one_hot = F.one_hot(target, num_classes=num_classes).float()
    # Calculate MAE loss (L1 loss) per sample
    # Sum the absolute differences across all classes for each sample
    loss_per_sample = F.l1_loss(probs, target_one_hot, reduction='none').sum(dim=1)
    # Optional: Normalize by number of classes (or num_classes - 1)
    loss = loss_per_sample / num_classes
    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# --- Function to select and calculate the desired loss ---
def calculate_loss(logits, target, loss_type, gamma=GAMMA, beta=BETA, num_classes=NUM_CLASSES):
    """Calculates the specified loss."""
    if loss_type == 'CE':
        # Use the predefined PyTorch CE loss
        return criteria(logits, target)
    elif loss_type == 'FL':
        # Instantiate and use our FocalLoss class
        criterion_fl = FocalLoss(gamma=gamma)
        return criterion_fl(logits, target)
    elif loss_type == 'NCE':
        # Use our NCE function
        return nce_loss(logits, target)
    elif loss_type == 'NFL':
        # Use our NFL function
        return nfl_loss(logits, target, gamma=gamma)
    elif loss_type == 'APL-NCE-MAE':
        # Combine NCE (Active) and MAE (Passive)
        l_active = nce_loss(logits, target)
        l_passive = mae_loss_passive(logits, target, num_classes=num_classes)
        return l_active + beta * l_passive # Weighted sum
    elif loss_type == 'APL-NFL-MAE':
        # Combine NFL (Active) and MAE (Passive)
        l_active = nfl_loss(logits, target, gamma=gamma)
        l_passive = mae_loss_passive(logits, target, num_classes=num_classes)
        return l_active + beta * l_passive # Weighted sum
    else:
        # Error handling for unknown loss types
        raise ValueError(f"Unknown loss type: {loss_type}")
    


    # --- Training Function (One Epoch) ---
def train_one_epoch(model, train_loader, optimizer, loss_type, device):
    """ Trains the model for one full pass through the training data. """
    model.train() # Set the model to training mode (enables dropout, batch norm updates)
    total_loss = 0.0
    # Loop through batches of data provided by the train_loader
    for inputs, labels in train_loader:
        # Move data to the designated device (GPU or CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # --- Forward Pass ---
        outputs = model(inputs) # Get model predictions (logits)

        # --- Calculate Loss ---
        # Use the helper function to compute the loss based on the selected type
        loss = calculate_loss(outputs, labels, loss_type, gamma=GAMMA, beta=BETA, num_classes=NUM_CLASSES)

        # --- Backward Pass and Optimization ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss for reporting average loss for the epoch
        total_loss += loss.item() * inputs.size(0) # loss.item() gets scalar value, multiply by batch size

    # Return average loss for the epoch
    return total_loss / len(train_loader.dataset)

# --- Evaluation Function ---
def evaluate(model, test_loader, device):
    """ Evaluates the model on the clean test dataset. """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        # Loop through batches in the test loader
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device) # Move data to device

            # --- Forward Pass ---
            outputs = model(inputs) # Get model predictions (logits)

            # --- Get Predictions ---
            # Find the class index with the highest logit score for each input
            _, predicted = torch.max(outputs.data, 1)

            # --- Calculate Accuracy ---
            total += labels.size(0) # Count total samples
            correct += (predicted == labels).sum().item() # Count correct predictions

    # Return overall accuracy
    accuracy = 100 * correct / total
    return accuracy



# --- Main Experiment Loop ---
for loss_fn in LOSS_FUNCTIONS_TO_TEST:
    # Iterate through each noise rate for the current loss function
    for noise_rate in SYMMETRIC_NOISE_RATES:
        print(f"\n===== Training Configuration ===== ")
        print(f"Loss Function: {loss_fn}")
        print(f"Symmetric Noise Rate: {noise_rate}")
        print(f"==================================")

        # --- Prepare Data ---
        current_trainset = noisy_trainsets[noise_rate]
        train_loader = torch.utils.data.DataLoader(
            current_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True) # pin_memory might speed up GPU transfer

        # --- Initialize Model and Optimizer ---
        # Create a FRESH model instance for each experiment run to ensure fair comparison
        current_model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
        # Define the optimizer (Adam is a good default)
        optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE)
        # Optional: Define a learning rate scheduler
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) # Example: reduce LR every 20 epochs

        # --- Training and Evaluation Loop for this Configuration ---
        best_test_acc = 0.0
        for epoch in range(EPOCHS):
            # Train for one epoch
            train_loss = train_one_epoch(current_model, train_loader, optimizer, loss_fn, DEVICE)
            # Evaluate on the CLEAN test set
            test_acc = evaluate(current_model, test_loader, DEVICE)

            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

            # Keep track of the best test accuracy achieved during this run
            if test_acc > best_test_acc:
                best_test_acc = test_acc

            # if scheduler: scheduler.step()

        # --- Store Result ---
        RESULTS[(loss_fn, noise_rate)] = best_test_acc
        print(f"--- Finished Training ---")
        print(f"Loss: {loss_fn}, Noise: {noise_rate} => Best Test Accuracy: {best_test_acc:.2f}%")
        print(f"-------------------------")


# --- Display Final Results Table ---
print("\n===== Final Experiment Results Summary =====")
for loss_fn in LOSS_FUNCTIONS_TO_TEST:
    acc_row = [f"{RESULTS.get((loss_fn, eta), 0.0):.2f}" for eta in SYMMETRIC_NOISE_RATES]
    print(f"Loss: {loss_fn:<15} | Accuracies @ η={SYMMETRIC_NOISE_RATES}: {acc_row}")
print("==========================================")





# --- Plotting Results ---
plt.figure(figsize=(10, 7)) # Create a figure to hold the plot

# Iterate through each loss function to plot its performance curve
for loss_fn in LOSS_FUNCTIONS_TO_TEST:
    # Extract the accuracies for this loss function across all tested noise rates
    accuracies = [RESULTS.get((loss_fn, eta), 0.0) for eta in SYMMETRIC_NOISE_RATES] # Use .get for safety if a run failed
    # Plot noise rates (x-axis) vs accuracies (y-axis)
    plt.plot(SYMMETRIC_NOISE_RATES, accuracies, marker='o', linestyle='-', label=loss_fn)

# --- Add Plot Decorations ---
plt.title('Model Test Accuracy vs. Symmetric Noise Rate (CIFAR-10)')
plt.xlabel('Symmetric Noise Rate (η)') 
plt.ylabel('Best Test Accuracy (%)') 
plt.xticks(SYMMETRIC_NOISE_RATES)
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)

plt.show()
