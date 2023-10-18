import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Subset
import flwr as fl
import torch.nn.functional as F
from torchvision.transforms import Compose,ToTensor,Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim

# Set the number of clients for each dataset
NUM_CLIENTS_CIFAR10 = 12
NUM_CLIENTS_EMNIST = 12
NUM_CLIENTS_SVHN = 12
NUM_CLIENTS_FASHION_MNIST = 12
NUM_EPOCHS = 1
BATCH_SIZE = 100
LEARNING_RATE = 0.02

# Define a function to load CIFAR-10 dataset and split it for each client
def load_cifar10_for_clients(num_clients):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./datacifar10', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./datacifar10', train=False, download=True, transform=transform)

    # Split the dataset equally among clients
    total_train_samples = len(train_dataset)
    samples_per_client = total_train_samples // num_clients

    # Create a list to store the train and test datasets for each client
    client_train_datasets = []
    client_test_datasets = []

    # Split train and test datasets for each client
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client

        train_subset = Subset(train_dataset, list(range(start_idx, end_idx)))
        test_subset = Subset(test_dataset, list(range(start_idx, end_idx)))

        client_train_datasets.append(train_subset)
        client_test_datasets.append(test_subset)

    return client_train_datasets, client_test_datasets

# Define a function to load EMNIST dataset and split it for each client starting from client_13
def load_emnist_for_clients(num_clients, start_client_num):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load EMNIST dataset
    train_dataset = torchvision.datasets.EMNIST(root='./dataemnist', split='balanced', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.EMNIST(root='./dataemnist', split='balanced', train=False, download=True, transform=transform)

    # Split the dataset equally among clients starting from start_client_num
    total_train_samples = len(train_dataset)
    samples_per_client = total_train_samples // num_clients

    # Create a list to store the train and test datasets for each client
    client_train_datasets = []
    client_test_datasets = []

    # Split train and test datasets for each client
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client

        train_subset = Subset(train_dataset, list(range(start_idx, end_idx)))
        test_subset = Subset(test_dataset, list(range(start_idx, end_idx)))

        client_train_datasets.append(train_subset)
        client_test_datasets.append(test_subset)

    return client_train_datasets, client_test_datasets


# Define a function to load SVHN dataset and split it for each client
def load_svhn_for_clients(num_clients):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load SVHN dataset
    train_dataset = torchvision.datasets.SVHN(root='./datasvhn', split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.SVHN(root='./datasvhn', split='test', download=True, transform=transform)

    # Split the dataset equally among clients
    total_train_samples = len(train_dataset)
    samples_per_client = total_train_samples // num_clients

    # Create a list to store the train and test datasets for each client
    client_train_datasets = []
    client_test_datasets = []

    # Split train and test datasets for each client
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client

        train_subset = Subset(train_dataset, list(range(start_idx, end_idx)))
        test_subset = Subset(test_dataset, list(range(start_idx, end_idx)))

        client_train_datasets.append(train_subset)
        client_test_datasets.append(test_subset)

    return client_train_datasets, client_test_datasets



# Define a function to load Fashion MNIST dataset and split it for each client
def load_fashion_mnist_for_clients(num_clients):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load Fashion MNIST dataset
    train_dataset = torchvision.datasets.FashionMNIST(root='./datafashion', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./datafashion', train=False, download=True, transform=transform)

    # Split the dataset equally among clients
    total_train_samples = len(train_dataset)
    samples_per_client = total_train_samples // num_clients

    # Create a list to store the train and test datasets for each client
    client_train_datasets = []
    client_test_datasets = []

    # Split train and test datasets for each client
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client

        train_subset = Subset(train_dataset, list(range(start_idx, end_idx)))
        test_subset = Subset(test_dataset, list(range(start_idx, end_idx)))

        client_train_datasets.append(train_subset)
        client_test_datasets.append(test_subset)

    return client_train_datasets, client_test_datasets


# Load CIFAR-10 datasets for each client
cifar10_train_datasets, cifar10_test_datasets = load_cifar10_for_clients(NUM_CLIENTS_CIFAR10)

# Load EMNIST datasets for each client starting from client_13
emnist_train_datasets, emnist_test_datasets = load_emnist_for_clients(NUM_CLIENTS_EMNIST, start_client_num=13)

# Load SVHN datasets for each client
svhn_train_datasets, svhn_test_datasets = load_svhn_for_clients(NUM_CLIENTS_SVHN)

# Load Fashion MNIST datasets for each client
fashion_mnist_train_datasets, fashion_mnist_test_datasets = load_fashion_mnist_for_clients(NUM_CLIENTS_FASHION_MNIST)

class CustomCIFAR10Model(nn.Module):
    def __init__(self):
        super(CustomCIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomEMNISTModel(nn.Module):
    def __init__(self):
        super(CustomEMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 47)  # EMNIST has 47 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomSVHNModel(nn.Module):
    def __init__(self):
        super(CustomSVHNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # SVHN has 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class CustomFashionMNISTModel(nn.Module):
    def __init__(self):
        super(CustomFashionMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # Fashion MNIST has 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Helper function to train the model and evaluate accuracy
def train_model(model, train_loader, loss_function, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%")
    
    return model, epoch_loss, epoch_accuracy



# Create a list of clients with names like client_1, client_2, ...
clients = []

# Append CIFAR-10 clients
for i in range(NUM_CLIENTS_CIFAR10):
    client = {
        "name": f"client_{i+1}",
        "train_data_cifar10": cifar10_train_datasets[i],
        "test_data_cifar10": cifar10_test_datasets[i],
        "model": CustomCIFAR10Model()
    }
    clients.append(client)
    


# Append EMNIST clients
for i in range(NUM_CLIENTS_EMNIST):
    client = {
        "name": f"client_{i+NUM_CLIENTS_CIFAR10}",
        "train_data_emnist": emnist_train_datasets[i],
        "test_data_emnist": emnist_test_datasets[i],
        "model": CustomEMNISTModel()
    }
    clients.append(client)


# Append SVHN clients
for i in range(NUM_CLIENTS_SVHN):
    client = {
        "name": f"client_{i+NUM_CLIENTS_CIFAR10+NUM_CLIENTS_EMNIST}",
        "train_data_svhn": svhn_train_datasets[i],
        "test_data_svhn": svhn_test_datasets[i],
        "model": CustomSVHNModel()
    }
    clients.append(client)

# Append Fashion MNIST clients
for i in range(NUM_CLIENTS_FASHION_MNIST):
    client = {
        "name": f"client_{i+NUM_CLIENTS_CIFAR10+NUM_CLIENTS_EMNIST+NUM_CLIENTS_SVHN}",
        "train_data_fashion_mnist": fashion_mnist_train_datasets[i],
        "test_data_fashion_mnist": fashion_mnist_test_datasets[i],
        "model": CustomFashionMNISTModel()
    }
    clients.append(client)

# Example to access train data for client_1 for CIFAR-10
#client_1_train_data_cifar10 = clients[0]["train_data_cifar10"]

# Example to access train data for client_13 for EMNIST
#client_13_train_data_emnist = clients[12]["train_data_emnist"]



# Print the number of samples for train and test datasets for clients 
for i in range(12):
    client_cifar = clients[i]  # Access EMNIST clients starting from index 1
    print(f"{client_cifar['name']}: Train samples (CIFAR) - {len(client_cifar['train_data_cifar10'])}, Test samples (CIFAR) - {len(client_cifar['test_data_cifar10'])}")
for i in range(12):
    client_emnist = clients[i + NUM_CLIENTS_CIFAR10]  # Access EMNIST clients starting from index 13
    print(f"{client_emnist['name']}: Train samples (EMNIST) - {len(client_emnist['train_data_emnist'])}, Test samples (EMNIST) - {len(client_emnist['test_data_emnist'])}")
for i in range(12):
    client_svhn = clients[i + + NUM_CLIENTS_CIFAR10 + NUM_CLIENTS_EMNIST]  # Access svhn clients starting from index 26
    print(f"{client_svhn['name']}: Train samples (SVHN) - {len(client_svhn['train_data_svhn'])}, Test samples (SVHN) - {len(client_svhn['test_data_svhn'])}")
for i in range(12):
    client_fashion = clients[i + NUM_CLIENTS_CIFAR10 + NUM_CLIENTS_EMNIST + NUM_CLIENTS_SVHN]  # Access svhn clients starting from index 37
    print(f"{client_fashion['name']}: Train samples (FASHION_MNIST) - {len(client_fashion['train_data_fashion_mnist'])}, Test samples (FASHION_MNIST) - {len(client_fashion['test_data_fashion_mnist'])}")

# Initialize the list to store loss and accuracy for each client
client_losses = [[] for _ in range(NUM_CLIENTS_CIFAR10 + NUM_CLIENTS_EMNIST + NUM_CLIENTS_SVHN + NUM_CLIENTS_FASHION_MNIST)]
client_accuracies = [[] for _ in range(NUM_CLIENTS_CIFAR10 + NUM_CLIENTS_EMNIST + NUM_CLIENTS_SVHN + NUM_CLIENTS_FASHION_MNIST)]


# Training and evaluation loop for CIFAR-10 clients
for i in range(NUM_CLIENTS_CIFAR10):
    train_loader = DataLoader(cifar10_train_datasets[i], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(cifar10_test_datasets[i], batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model, loss function, and optimizer for CIFAR-10
    model = CustomCIFAR10Model()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    model, loss, accuracy = train_model(model, train_loader, loss_function, optimizer, NUM_EPOCHS)
    
    # Store loss and accuracy along with client name
    client_losses[i].append((clients[i]["name"], loss))
    client_accuracies[i].append((clients[i]["name"], accuracy))

# Training and evaluation loop for EMNIST clients
for i in range(NUM_CLIENTS_EMNIST):
    train_loader = DataLoader(emnist_train_datasets[i], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(emnist_test_datasets[i], batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model, loss function, and optimizer for EMNIST
    model = CustomEMNISTModel()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    model, loss, accuracy = train_model(model, train_loader, loss_function, optimizer, NUM_EPOCHS)
    
    # Store loss and accuracy
    client_losses[NUM_CLIENTS_CIFAR10 + i].append((clients[NUM_CLIENTS_CIFAR10 + i]["name"], loss))
    client_accuracies[NUM_CLIENTS_CIFAR10 + i].append((clients[NUM_CLIENTS_CIFAR10 + i]["name"], accuracy))

# Training and evaluation loop for SVHN clients
for i in range(NUM_CLIENTS_SVHN):
    train_loader = DataLoader(svhn_train_datasets[i], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(svhn_test_datasets[i], batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model, loss function, and optimizer for SVHN
    model = CustomSVHNModel()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    model, loss, accuracy = train_model(model, train_loader, loss_function, optimizer, NUM_EPOCHS)
    
    # Store loss and accuracy
    client_losses[NUM_CLIENTS_CIFAR10 + NUM_CLIENTS_EMNIST + i].append((clients[NUM_CLIENTS_CIFAR10 + NUM_CLIENTS_EMNIST + i]["name"], loss))
    client_accuracies[NUM_CLIENTS_CIFAR10 + NUM_CLIENTS_EMNIST + i].append((clients[NUM_CLIENTS_CIFAR10 + NUM_CLIENTS_EMNIST + i]["name"], accuracy))

# Training and evaluation loop for Fashion MNIST clients
for i in range(NUM_CLIENTS_FASHION_MNIST):
    train_loader = DataLoader(fashion_mnist_train_datasets[i], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(fashion_mnist_test_datasets[i], batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model, loss function, and optimizer for Fashion MNIST
    model = CustomFashionMNISTModel()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    model, loss, accuracy = train_model(model, train_loader, loss_function, optimizer, NUM_EPOCHS)
    
    # Store loss and accuracy
    # Store loss and accuracy along with client name
    client_losses[NUM_CLIENTS_CIFAR10 + NUM_CLIENTS_EMNIST + NUM_CLIENTS_SVHN + i].append((clients[NUM_CLIENTS_CIFAR10 + NUM_CLIENTS_EMNIST + NUM_CLIENTS_SVHN + i]["name"], loss))
    client_accuracies[NUM_CLIENTS_CIFAR10 + NUM_CLIENTS_EMNIST + NUM_CLIENTS_SVHN + i].append((clients[NUM_CLIENTS_CIFAR10 + NUM_CLIENTS_EMNIST + NUM_CLIENTS_SVHN + i]["name"], accuracy))


# Print client losses and accuracies with client names
print("Client Losses:")
for i, client_loss in enumerate(client_losses):
    print(f"Client {clients[i]['name']} - {client_loss}")

print("Client Accuracies:")
for i, client_accuracy in enumerate(client_accuracies):
    print(f"Client {clients[i]['name']} - {client_accuracy}")
