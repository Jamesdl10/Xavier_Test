
# ========= Import Nesscary Libraries =========# 


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# ========== Script Functions  ================# 

#%% Classes and Functions ##########################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels1, output_channels1, kernel_size=kernel_size1)
        self.conv2 = nn.Conv2d(input_channels2, output_channels2, kernel_size=kernel_size2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(input_features1, output_features1)
        self.fc2 = nn.Linear(input_features2, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, input_features1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(epoch):
    network.train() # need to use this as we are now using dropout
    log_interval = 10
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
            (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
        
def test():
    network.eval() # need to use these as we are now using dropout
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def classify(data, threshold):
    # Get the model's output
    output = network(data)
    
    # Convert the log-probabilities to probabilities
    prob = torch.exp(output)
    
    # Get the maximum probabilities and their corresponding indices (class predictions)
    max_prob, pred = torch.max(prob, dim=1)
    
    # Find the indices of the predictions which have probabilities above the threshold
    inds_classified = (max_prob > threshold).nonzero(as_tuple=True)[0]
    
    return pred, max_prob, inds_classified


# ===========  3 Data import and preprocessing =============# 

# Define the path for storing dataset and download the MNIST dataset
files = './files/'
train_data = torchvision.datasets.MNIST(files, train=True, download=True)
# test_data = torchvision.datasets.MNIST(files, train=False, download=True) # Uncomment to download the test data if needed

# Convert the dataset to float and calculate the mean and standard deviation of pixel values
data = train_data.data.type(torch.float)
data_mean = data.mean().item()  # Mean of pixel values
data_std = data.std().item()  # Standard deviation of pixel values
print("Mean:", data_mean)
print("Standard Deviation:", data_std)

# Set batch sizes for training and testing
batch_size_train = 64
batch_size_test = 1000

# Create data loaders with normalization using the calculated mean and standard deviation
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(files, train=True, download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # Convert images to tensors
        torchvision.transforms.Normalize((data_mean / 255,), (data_std / 255,))  # Normalize the data
    ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(files, train=False, download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # Convert images to tensors
        torchvision.transforms.Normalize((data_mean / 255,), (data_std / 255,))  # Normalize the data
    ])),
    batch_size=batch_size_test, shuffle=True)

# Generate an example batch from the training data
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

# Print the shape of example data
print(f"Shape of example data: {example_data.shape}")  # Shape will be [batch_size, channels, height, width]

# Visualize some example images from the training batch
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')  # Display the image in grayscale
    plt.title(f"Ground Truth: {example_targets[i]}")  # Display the label
    plt.xticks([])  # Hide x-axis ticks
    plt.yticks([])  # Hide y-axis ticks
plt.show()

#========== 4 A CNN model ============= # 

# Define parameters for the convolutional layers
input_channels1 = 1  # Grayscale images have 1 channel
output_channels1 = 16  # First convolutional layer will output 16 channels
kernel_size1 = 5  # Kernel size of 5x5 for the first layer
input_channels2 = output_channels1  # Input to second layer is the output from the first layer
output_channels2 = 32  # Second convolutional layer will output 32 channels
kernel_size2 = 5  # Kernel size of 5x5 for the second layer

# Define convolutional layers
conv_layer1 = nn.Conv2d(input_channels1, output_channels1, kernel_size=kernel_size1)  # First convolutional layer
conv_layer2 = nn.Conv2d(input_channels2, output_channels2, kernel_size=kernel_size2)  # Second convolutional layer

# Pass example data through the first layer and print the output shape
h1 = conv_layer1(example_data)
print(h1.shape)

# Apply ReLU activation and max pooling to the output of the first layer
h2 = F.relu(F.max_pool2d(h1, 2))
print(h2.shape)

# Pass the result through the second convolutional layer
h3 = conv_layer2(h2)
print(h3.shape)

# Apply ReLU activation and max pooling to the output of the second layer
h4 = F.relu(F.max_pool2d(h3, 2))
print(h4.shape)

# Explain the relationship between input and output channels for the layers
print("The number of output channels from the first layer (output_channels1) directly determines,"
      " the number of input channels for the second layer (input_channels2). This is because each"
      " output channel from the first layer is treated as a separate feature map, and the next layer"
      " will take these feature maps as its input.")

# Explain how kernel size affects the output dimensions
print("The kernel size reduces the spatial dimensions (height and width) of the output feature maps"
      " after each convolution. Larger kernels reduce the output size more.")

# Determine the number of input features for the first fully connected layer
# This is the total number of elements in h4 after flattening (channels * width * height)
input_features1 = h4.shape[1] * h4.shape[2] * h4.shape[3]

# Define the number of output features for the first fully connected layer
output_features1 = 128  # You can choose this based on desired model complexity

# The input to the second fully connected layer is the output of the first fully connected layer
input_features2 = output_features1

# The output of the second fully connected layer corresponds to the number of classes (0-9 digits)
output_features2 = 10

# Reshape the output from the convolutional layers to pass it through the fully connected layers
h4 = h4.view(-1, input_features1)

# Define fully connected layers
fc1 = nn.Linear(input_features1, output_features1)  # First fully connected layer
fc2 = nn.Linear(input_features2, output_features2)  # Second fully connected layer

# Apply ReLU activation to the first fully connected layer
h5 = F.relu(fc1(h4))
print(h5.shape)

# Apply ReLU activation to the second fully connected layer
h6 = F.relu(fc2(h5))
print(h6.shape)

# Test the untrained model using the example batch
network = Net()  # Instantiate the CNN model
test_outputs = network(example_data)  # Pass the example data through the network
pred = test_outputs.data.max(1, keepdim=True)[1]  # Get the predicted class

# Compare the predicted labels with the ground truth to compute accuracy
correct = pred.eq(example_targets.view_as(pred)).sum().item()  # Count correct predictions
total = example_data.size(0)  # Total number of examples
accuracy = correct / total * 100  # Calculate accuracy

# Print the accuracy of the untrained model
print(f'Accuracy of the untrained model: {accuracy:.2f}%')

#=========== 5 Training and validation ==============# 

# Define the optimizer with a learning rate of 0.01 and momentum of 0.25
learning_rate = 0.01
momentum = 0.25
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

n_epochs = 5  # Set the number of training epochs
train_losses = []  # Initialize list to store training losses
train_counter = []  # Counter for tracking number of training samples seen
test_losses = []  # List to store test losses
test_counter = [i * len(train_loader.dataset) for i in range(1, n_epochs + 1)]  # Track samples at each epoch

# Train the network for the specified number of epochs
for epoch in range(1, n_epochs + 1):
    train(epoch)  # Call the train function for each epoch
    test()  # Evaluate the network on test data after each epoch

# Plot the training and test losses
plt.figure(figsize=(10,6))
plt.plot(train_counter, train_losses, label='Training Loss')  # Plot training loss
plt.plot(test_counter, test_losses, 'ro', label='Test Loss')  # Plot test loss in red circles

# Add labels, title, legend, and grid to the plot
plt.xlabel('Number of Samples Seen')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Time')
plt.legend(loc='upper right')
plt.grid(True)

# Display the plot
plt.show()

# Load a batch of test data for visualization
test_examples = enumerate(test_loader)
batch_idx, (test_data, test_targets) = next(test_examples)

# Pass the first 6 test samples through the trained network and get predictions
test_predictions = network(test_data[:6])
predicted_labels = test_predictions.data.max(1, keepdim=True)[1]  # Get predicted class

# Create a figure and subplots for displaying test images
fig, axs = plt.subplots(2, 3, figsize=(10, 7))

# Loop through the subplots and display images with their predicted labels
for i, ax in enumerate(axs.flatten()):
    # Display the test image
    ax.imshow(test_data[i][0], cmap='gray', interpolation='none')
    
    # Annotate the image with the predicted label
    ax.set_title(f"Predicted: {predicted_labels[i].item()}")
    
    # Remove the x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

#============== 6 Robust classification of the samples ================#

threshold = 0.99  # Set a threshold for classifying samples

# Initialize counters to keep track of classified samples and correct predictions
total_classified = 0
total_correct = 0
total_samples = 0

# Perform robust classification without computing gradients
with torch.no_grad():
    for data, target in test_loader:
        # Classify the test data using the defined threshold
        pred, prob, inds_classified = classify(data, threshold)
        
        # Count the number of classified samples and correct predictions
        total_classified += len(inds_classified)
        total_correct += pred[inds_classified].eq(target[inds_classified]).sum().item()
        
        # Update the total number of samples
        total_samples += len(data)
        
# Calculate the percentage of samples classified
percentage_classified = (total_classified / total_samples) * 100

# Calculate the accuracy on the classified samples
accuracy_classified = (total_correct / total_classified) * 100 if total_classified > 0 else 0

# Print the results
print(f"Chosen Threshold: {threshold}")
print(f"Percentage of samples classified: {percentage_classified:.2f}%")
print(f"Accuracy on classified samples: {accuracy_classified:.2f}%")


# Initialize lists to store unclassified samples, their predictions, and probabilities
unclassified_samples = []
unclassified_preds = []
unclassified_probs = []

# Set a counter for unclassified samples
count = 0

# Loop through the test data and find unclassified samples
with torch.no_grad():
    for data, target in test_loader:
        # Break the loop after finding 6 unclassified samples
        if count >= 6:
            break
        
        # Classify the data with the threshold
        pred, prob, inds_classified = classify(data, threshold)
        
        # Find unclassified samples (those with probabilities below the threshold)
        inds_unclassified = (prob <= threshold).nonzero(as_tuple=True)[0]
        
        # Store the unclassified samples and their predictions
        for index in inds_unclassified:
            unclassified_samples.append(data[index])
            unclassified_preds.append(pred[index].item())
            unclassified_probs.append(prob[index].max().item())
            count += 1
            
            # Stop after 6 unclassified samples
            if count >= 6:
                break

# Create subplots to display the unclassified images and their predictions
fig, axs = plt.subplots(2, 3, figsize=(10, 6))

# Loop through the subplots and display the unclassified images
for i, ax in enumerate(axs.flatten()):
    # Display the image
    ax.imshow(unclassified_samples[i][0], cmap='gray', interpolation='none')
    
    # Annotate the image with the predicted label and probability
    ax.set_title(f"Pred: {unclassified_preds[i]}, Prob: {unclassified_probs[i]:.3f}")
    
    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
