import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# RBM Hyperparameters
N_VISIBLE = 784  # 28x28 MNIST pixels
N_HIDDEN = 512   # Number of hidden p-bits/neurons. Try 512, 256, or even 64 to test parameter efficiency
K_CD = 1         # Number of Gibbs sampling steps in Contrastive Divergence
RBM_EPOCHS = 20
RBM_BATCH_SIZE = 64
RBM_LEARNING_RATE = 0.01 # Adjusted from 0.1, common for RBMs

# Classifier Hyperparameters
CLASSIFIER_EPOCHS = 30
CLASSIFIER_BATCH_SIZE = 128 # Can be larger for classifier
CLASSIFIER_LEARNING_RATE = 0.001

# --- MNIST Data Loading ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(N_VISIBLE)), # Flatten
    transforms.Lambda(lambda x: torch.bernoulli(x)) # Binarize MNIST pixels
])

trainset_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# For RBM, we only need the images, not labels for unsupervised training
rbm_trainloader = DataLoader(trainset_full, batch_size=RBM_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True if DEVICE.type == 'cuda' else False)

# For classifier training, we need images and labels
# We can split trainset_full if needed for validation, or just use it all for classifier training
classifier_trainloader = DataLoader(trainset_full, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True if DEVICE.type == 'cuda' else False)
testloader = DataLoader(testset, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True if DEVICE.type == 'cuda' else False)


# --- RBM Implementation ---
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.1) # Weight matrix
        self.v_bias = nn.Parameter(torch.zeros(n_visible)) # Visible layer bias
        self.h_bias = nn.Parameter(torch.zeros(n_hidden)) # Hidden layer bias

    def sample_h_given_v(self, v):
        # P(h_j=1 | v) = sigmoid(W_j * v + c_j)
        # v is (batch_size, n_visible)
        # self.W is (n_hidden, n_visible)
        # h_activation will be (batch_size, n_hidden)
        h_activation = torch.sigmoid(torch.matmul(v, self.W.T) + self.h_bias)
        # Sample h from P(h|v) (Bernoulli distribution)
        h_sample = torch.bernoulli(h_activation)
        return h_sample, h_activation

    def sample_v_given_h(self, h):
        # P(v_i=1 | h) = sigmoid(W_i * h + b_i)
        # h is (batch_size, n_hidden)
        # self.W.T is (n_visible, n_hidden)
        # v_activation will be (batch_size, n_visible)
        v_activation = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        # Sample v from P(v|h)
        v_sample = torch.bernoulli(v_activation)
        return v_sample, v_activation

    def forward(self, v0):
        # Positive phase: from data v0, sample h0
        h0_sample, _ = self.sample_h_given_v(v0)

        # Negative phase (Contrastive Divergence CD-k)
        # Start Gibbs sampling from h0_sample
        hn_sample = h0_sample
        for _ in range(K_CD):
            vn_sample, _ = self.sample_v_given_h(hn_sample)
            hn_sample, _ = self.sample_h_given_v(vn_sample)
        
        # We don't need the final vn_sample for gradient calculation with CD-k,
        # but we need the activations that led to it.
        # For CD-k, the "model's expectation" is derived from vn_sample and hn_sample (from last Gibbs step).
        # Specifically, it's <v_k h_k>_model
        # v_k is vn_sample from the last step.
        # h_k_activation is P(h|v_k) using vn_sample.
        _, vk_model_activation = self.sample_v_given_h(hn_sample) # This is v_k
        _, hk_model_activation = self.sample_h_given_v(vk_model_activation) # P(h|v_k)
        
        return v0, h0_sample, vk_model_activation, hk_model_activation


# --- RBM Training ---
print(f"\nStarting RBM Training ({N_VISIBLE} visible, {N_HIDDEN} hidden, {K_CD}-step CD)")
rbm = RBM(N_VISIBLE, N_HIDDEN).to(DEVICE)
# Parameters for RBM: W (N_HIDDEN * N_VISIBLE) + v_bias (N_VISIBLE) + h_bias (N_HIDDEN)
rbm_params = N_HIDDEN * N_VISIBLE + N_VISIBLE + N_HIDDEN
print(f"RBM Parameters: {rbm_params}")

# Optimizer for RBM - manual gradient updates are common, but Adam can work
# For manual:
# optimizer = None 
# For Adam (less common for pure RBMs, but can work):
# optimizer = optim.Adam(rbm.parameters(), lr=RBM_LEARNING_RATE)

start_time = time.time()
for epoch in range(RBM_EPOCHS):
    epoch_loss = 0.0
    for i, (batch_v0, _) in enumerate(rbm_trainloader):
        batch_v0 = batch_v0.to(DEVICE)

        # Get positive and negative phase samples/activations
        v0_data, h0_data_sample, vk_model_recon, hk_model_activation = rbm(batch_v0)
        # v0_data: original input data (batch_size, n_visible)
        # h0_data_sample: hidden samples from v0_data (batch_size, n_hidden)
        # vk_model_recon: reconstructed visible units after K steps (batch_size, n_visible)
        # hk_model_activation: hidden activations from vk_model_recon P(h|v_k) (batch_size, n_hidden)

        # Calculate gradients for CD-k (manual update)
        # <v_i h_j>_data
        positive_grad_W = torch.matmul(h0_data_sample.T, v0_data)
        # <v_i h_j>_model
        negative_grad_W = torch.matmul(hk_model_activation.T, vk_model_recon) # Use activations for model expectation

        # Update rule: W_new = W_old + learning_rate * (<vh>_data - <vh>_model) / batch_size
        # Note: PyTorch nn.Parameter.data allows direct manipulation
        with torch.no_grad(): # Ensure these operations are not tracked for autograd
            # W update: (n_hidden, n_visible)
            grad_W = (positive_grad_W - negative_grad_W) / batch_v0.size(0)
            rbm.W.data += RBM_LEARNING_RATE * grad_W
            
            # Visible bias update: (n_visible)
            # grad_v_bias = <v>_data - <v>_model
            grad_v_bias = torch.sum(v0_data - vk_model_recon, dim=0) / batch_v0.size(0)
            rbm.v_bias.data += RBM_LEARNING_RATE * grad_v_bias

            # Hidden bias update: (n_hidden)
            # grad_h_bias = <h>_data - <h>_model
            # For <h>_data, use P(h|v0), not the sample. Sample h0 from P(h|v0)
            _, h0_data_activation = rbm.sample_h_given_v(v0_data)
            grad_h_bias = torch.sum(h0_data_activation - hk_model_activation, dim=0) / batch_v0.size(0)
            rbm.h_bias.data += RBM_LEARNING_RATE * grad_h_bias

        # Reconstruction error (for monitoring, not direct loss for CD-k parameter update)
        loss = torch.mean((v0_data - vk_model_recon)**2)
        epoch_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{RBM_EPOCHS}], Batch [{i+1}/{len(rbm_trainloader)}], Recon Loss: {loss.item():.4f}")
    
    print(f"Epoch [{epoch+1}/{RBM_EPOCHS}] completed. Average Recon Loss: {epoch_loss/len(rbm_trainloader):.4f}")

    # Optional: Visualize RBM weights or reconstructions
    if (epoch + 1) % 5 == 0:
        # Visualize some reconstructions
        rbm.eval() # Set to eval mode if it had dropout/batchnorm, not critical for this RBM
        with torch.no_grad():
            sample_v = next(iter(testloader))[0][:16].to(DEVICE) # Get 16 samples
            _, h_sample = rbm.sample_h_given_v(sample_v)
            v_reconstructed, _ = rbm.sample_v_given_h(h_sample) # Reconstruct once
            
            fig, axes = plt.subplots(4, 8, figsize=(12, 6))
            for k in range(16):
                row, col_orig = divmod(k, 4)
                axes[row, col_orig*2].imshow(sample_v[k].cpu().reshape(28, 28), cmap='gray')
                axes[row, col_orig*2].set_title("Original")
                axes[row, col_orig*2].axis('off')
                axes[row, col_orig*2+1].imshow(v_reconstructed[k].cpu().reshape(28, 28), cmap='gray')
                axes[row, col_orig*2+1].set_title("Recon")
                axes[row, col_orig*2+1].axis('off')
            plt.suptitle(f"RBM Reconstructions Epoch {epoch+1}")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f"rbm_reconstructions_epoch_{epoch+1}.png")
            plt.close(fig)
        rbm.train() # Back to train mode

end_time = time.time()
print(f"RBM training finished. Time: {end_time - start_time:.2f} seconds.")
# Save the RBM (optional)
# torch.save(rbm.state_dict(), "rbm_mnist.pth")
# rbm.load_state_dict(torch.load("rbm_mnist.pth"))

# --- Classifier on RBM Features ---
class Classifier(nn.Module):
    def __init__(self, n_input, n_output):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.linear(x) # Output logits

print(f"\nStarting Classifier Training (Input: {N_HIDDEN} RBM features, Output: 10 classes)")
classifier = Classifier(N_HIDDEN, 10).to(DEVICE) # 10 classes for MNIST
# Parameters for Classifier: linear_weights (N_HIDDEN * 10) + linear_bias (10)
classifier_params = N_HIDDEN * 10 + 10
print(f"Classifier Parameters: {classifier_params}")
print(f"Total Parameters (RBM + Classifier): {rbm_params + classifier_params}")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=CLASSIFIER_LEARNING_RATE)

# Freeze RBM weights - we are using it as a fixed feature extractor
rbm.eval()
for param in rbm.parameters():
    param.requires_grad = False

start_time = time.time()
for epoch in range(CLASSIFIER_EPOCHS):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, (inputs, labels) in enumerate(classifier_trainloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        # Get RBM hidden activations as features (use probabilities, not samples, for stability)
        with torch.no_grad():
            _, hidden_features = rbm.sample_h_given_v(inputs) # Using h_activation

        outputs = classifier(hidden_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        if (i + 1) % 100 == 0:
            print(f"Classifier Epoch [{epoch+1}/{CLASSIFIER_EPOCHS}], Batch [{i+1}/{len(classifier_trainloader)}], Loss: {loss.item():.4f}")
    
    train_accuracy = 100 * correct_train / total_train
    print(f"Classifier Epoch [{epoch+1}/{CLASSIFIER_EPOCHS}] completed. Avg Loss: {running_loss/len(classifier_trainloader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Validation / Test
    correct_test = 0
    total_test = 0
    classifier.eval() # Set classifier to evaluation mode
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            _, hidden_features = rbm.sample_h_given_v(inputs)
            outputs = classifier(hidden_features)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct_test / total_test
    print(f"Classifier Epoch [{epoch+1}/{CLASSIFIER_EPOCHS}], Test Accuracy: {test_accuracy:.2f}%")
    classifier.train() # Back to train mode

end_time = time.time()
print(f"Classifier training finished. Time: {end_time - start_time:.2f} seconds.")

print("\n--- Final Evaluation on Test Set ---")
correct = 0
total = 0
rbm.eval()
classifier.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        _, hidden_features = rbm.sample_h_given_v(inputs)
        outputs = classifier(hidden_features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = 100 * correct / total
print(f'Accuracy of the network on the {total} test images: {final_accuracy:.2f} %')

if final_accuracy > 90:
    print("Achieved over 90% accuracy! Congratulations!")
elif final_accuracy > 85:
    print("Good result! Close to 90%. Further tuning might get you there.")
else:
    print("Decent start. Try tuning hyperparameters (N_HIDDEN, learning rates, epochs, K_CD) or a DBM.")