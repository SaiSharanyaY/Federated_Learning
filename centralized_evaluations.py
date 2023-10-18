import client_generation as cg
CLIENT_NUM_SELECTED = 8
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import Compose,ToTensor,Normalize
from torch.utils.data import DataLoader

# Flatten the list of client losses and sort by loss
all_client_losses = [loss for client_losses in cg.client_losses for _, loss in client_losses]
sorted_clients_by_loss = sorted(zip(cg.clients, all_client_losses), key=lambda x: x[1])

# Select the 8 clients with the least loss
selected_clients = [client for client, _ in sorted_clients_by_loss[:CLIENT_NUM_SELECTED]]

# Print the names of the selected clients and their corresponding loss
print("Selected Clients and Losses:")
for client, loss in sorted_clients_by_loss[:CLIENT_NUM_SELECTED]:
    print(f"{client['name']} - Loss: {loss}")

# Optionally, you can also print the names of all clients and their losses for reference
print("All Clients and Losses:")
for client, loss in sorted_clients_by_loss:
    print(f"Client {client['name']} - Loss: {loss}")


# Define a custom PyTorchClient class that inherits from flwr.Client
class CustomPyTorchClient(fl.client.Client):
    def __init__(self, model, train_loader, test_loader):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self):
        return [param.cpu().numpy() for param in self.model.parameters()]

    def fit(self, parameters, config):
        # Set the received parameters to the model
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.from_numpy(new_param)

        # Train the model
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

        return len(self.train_loader), {}

    def evaluate(self, parameters, config):
        # Set the received parameters to the model
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.from_numpy(new_param)

        # Evaluate the model
        self.model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.round().long()
                correct += pred.eq(target.view_as(pred)).sum().item()

        return len(self.test_loader), test_loss, correct / len(self.test_loader.dataset)


class CustomFlowerClient(fl.client.NumPyClient):
    def __init__(self, pytorch_client, name):
        self.pytorch_client = pytorch_client
        self.name = name

    def get_parameters(self):
        return self.pytorch_client.get_parameters()

    def set_parameters(self, parameters):
        self.pytorch_client.set_parameters(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        return self.pytorch_client.fit(parameters, config)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return self.pytorch_client.evaluate(parameters, config)
    


# Define a dictionary to map client names to corresponding model types and datasets
client_mapping = {
    'client_1': (cg.CustomCIFAR10Model, cg.clients[0]["train_data_cifar10"], cg.clients[0]["test_data_cifar10"]),
    'client_2': (cg.CustomCIFAR10Model, cg.clients[1]["train_data_cifar10"], cg.clients[1]["test_data_cifar10"]),
    'client_3': (cg.CustomCIFAR10Model, cg.clients[2]["train_data_cifar10"], cg.clients[2]["test_data_cifar10"]),
    'client_4': (cg.CustomCIFAR10Model, cg.clients[3]["train_data_cifar10"], cg.clients[3]["test_data_cifar10"]),
    'client_5': (cg.CustomCIFAR10Model, cg.clients[4]["train_data_cifar10"], cg.clients[4]["test_data_cifar10"]),
    'client_6': (cg.CustomCIFAR10Model, cg.clients[5]["train_data_cifar10"], cg.clients[5]["test_data_cifar10"]),
    'client_7': (cg.CustomCIFAR10Model, cg.clients[6]["train_data_cifar10"], cg.clients[6]["test_data_cifar10"]),
    'client_8': (cg.CustomCIFAR10Model, cg.clients[7]["train_data_cifar10"], cg.clients[7]["test_data_cifar10"]),
    'client_9': (cg.CustomCIFAR10Model, cg.clients[8]["train_data_cifar10"], cg.clients[8]["test_data_cifar10"]),
    'client_10': (cg.CustomCIFAR10Model, cg.clients[9]["train_data_cifar10"], cg.clients[9]["test_data_cifar10"]),
    'client_11': (cg.CustomCIFAR10Model, cg.clients[10]["train_data_cifar10"], cg.clients[10]["test_data_cifar10"]),
    'client_12': (cg.CustomCIFAR10Model, cg.clients[11]["train_data_cifar10"], cg.clients[11]["test_data_cifar10"]),
    'client_13': (cg.CustomEMNISTModel, cg.clients[12]["train_data_emnist"], cg.clients[12]["test_data_emnist"]),
    'client_14': (cg.CustomEMNISTModel, cg.clients[13]["train_data_emnist"], cg.clients[13]["test_data_emnist"]),
    'client_15': (cg.CustomEMNISTModel, cg.clients[14]["train_data_emnist"], cg.clients[14]["test_data_emnist"]),
    'client_16': (cg.CustomEMNISTModel, cg.clients[15]["train_data_emnist"], cg.clients[15]["test_data_emnist"]),
    'client_17': (cg.CustomEMNISTModel, cg.clients[16]["train_data_emnist"], cg.clients[16]["test_data_emnist"]),
    'client_18': (cg.CustomEMNISTModel, cg.clients[17]["train_data_emnist"], cg.clients[17]["test_data_emnist"]),
    'client_19': (cg.CustomEMNISTModel, cg.clients[18]["train_data_emnist"], cg.clients[18]["test_data_emnist"]),
    'client_20': (cg.CustomEMNISTModel, cg.clients[19]["train_data_emnist"], cg.clients[19]["test_data_emnist"]),
    'client_21': (cg.CustomEMNISTModel, cg.clients[20]["train_data_emnist"], cg.clients[20]["test_data_emnist"]),
    'client_22': (cg.CustomEMNISTModel, cg.clients[21]["train_data_emnist"], cg.clients[21]["test_data_emnist"]),
    'client_23': (cg.CustomEMNISTModel, cg.clients[22]["train_data_emnist"], cg.clients[22]["test_data_emnist"]),
    'client_24': (cg.CustomEMNISTModel, cg.clients[23]["train_data_emnist"], cg.clients[23]["test_data_emnist"]),
    'client_25': (cg.CustomSVHNModel, cg.clients[24]["train_data_svhn"], cg.clients[24]["test_data_svhn"]),
    'client_26': (cg.CustomSVHNModel, cg.clients[25]["train_data_svhn"], cg.clients[25]["test_data_svhn"]),
    'client_27': (cg.CustomSVHNModel, cg.clients[26]["train_data_svhn"], cg.clients[26]["test_data_svhn"]),
    'client_28': (cg.CustomSVHNModel, cg.clients[27]["train_data_svhn"], cg.clients[27]["test_data_svhn"]),
    'client_29': (cg.CustomSVHNModel, cg.clients[28]["train_data_svhn"], cg.clients[28]["test_data_svhn"]),
    'client_30': (cg.CustomSVHNModel, cg.clients[29]["train_data_svhn"], cg.clients[29]["test_data_svhn"]),
    'client_31': (cg.CustomSVHNModel, cg.clients[30]["train_data_svhn"], cg.clients[30]["test_data_svhn"]),
    'client_32': (cg.CustomSVHNModel, cg.clients[31]["train_data_svhn"], cg.clients[31]["test_data_svhn"]),
    'client_33': (cg.CustomSVHNModel, cg.clients[32]["train_data_svhn"], cg.clients[32]["test_data_svhn"]),
    'client_34': (cg.CustomSVHNModel, cg.clients[33]["train_data_svhn"], cg.clients[33]["test_data_svhn"]),
    'client_35': (cg.CustomSVHNModel, cg.clients[34]["train_data_svhn"], cg.clients[34]["test_data_svhn"]),
    'client_36': (cg.CustomSVHNModel, cg.clients[35]["train_data_svhn"], cg.clients[35]["test_data_svhn"]),
    'client_37': (cg.CustomFashionMNISTModel, cg.clients[36]["train_data_fashion_mnist"], cg.clients[36]["test_data_fashion_mnist"]),
    'client_38': (cg.CustomFashionMNISTModel, cg.clients[37]["train_data_fashion_mnist"], cg.clients[37]["test_data_fashion_mnist"]),
    'client_39': (cg.CustomFashionMNISTModel, cg.clients[38]["train_data_fashion_mnist"], cg.clients[38]["test_data_fashion_mnist"]),
    'client_40': (cg.CustomFashionMNISTModel, cg.clients[39]["train_data_fashion_mnist"], cg.clients[39]["test_data_fashion_mnist"]),
    'client_41': (cg.CustomFashionMNISTModel, cg.clients[40]["train_data_fashion_mnist"], cg.clients[40]["test_data_fashion_mnist"]),
    'client_42': (cg.CustomFashionMNISTModel, cg.clients[41]["train_data_fashion_mnist"], cg.clients[41]["test_data_fashion_mnist"]),
    'client_43': (cg.CustomFashionMNISTModel, cg.clients[42]["train_data_fashion_mnist"], cg.clients[42]["test_data_fashion_mnist"]),
    'client_44': (cg.CustomFashionMNISTModel, cg.clients[43]["train_data_fashion_mnist"], cg.clients[43]["test_data_fashion_mnist"]),
    'client_45': (cg.CustomFashionMNISTModel, cg.clients[44]["train_data_fashion_mnist"], cg.clients[44]["test_data_fashion_mnist"]),
    'client_46': (cg.CustomFashionMNISTModel, cg.clients[45]["train_data_fashion_mnist"], cg.clients[45]["test_data_fashion_mnist"]),
    'client_47': (cg.CustomFashionMNISTModel, cg.clients[46]["train_data_fashion_mnist"], cg.clients[46]["test_data_fashion_mnist"]),
    'client_48': (cg.CustomFashionMNISTModel, cg.clients[47]["train_data_fashion_mnist"], cg.clients[47]["test_data_fashion_mnist"]),
}

# Create variables dynamically based on the selected clients
for i, client in enumerate(selected_clients):
    client_name = client['name']
    if client_name in client_mapping:
        model_cls, train_data, test_data = client_mapping[client_name]

        # Create variables for the model and dataset
        model_variable_name = f"model_{client_name}"
        train_data_variable_name = f"train_data_{client_name}"
        test_data_variable_name = f"test_data_{client_name}"
        globals()[model_variable_name] = model_cls()
        globals()[train_data_variable_name] = train_data
        globals()[test_data_variable_name] = test_data

        # Create DataLoader instances
        train_loader_variable_name = f"train_loader_{client_name}"
        test_loader_variable_name = f"test_loader_{client_name}"
        globals()[train_loader_variable_name] = DataLoader(train_data, batch_size=cg.BATCH_SIZE, shuffle=True)
        globals()[test_loader_variable_name] = DataLoader(test_data, batch_size=cg.BATCH_SIZE, shuffle=False)

        # Create PyTorchClient instance
        pytorch_client_variable_name = f"pytorch_client_{client_name}"
        pytorch_client_instance = CustomPyTorchClient(
            model=globals()[model_variable_name],
            train_loader=globals()[train_loader_variable_name],
            test_loader=globals()[test_loader_variable_name]
        )
        globals()[pytorch_client_variable_name] = pytorch_client_instance

        #Create CustomFlowerClient instance
        custom_flower_client_variable_name = f"flower_client_{client_name}"
        custom_flower_client_instance = CustomFlowerClient(
        pytorch_client_instance, f"pytorch_client_{client_name}"
        )
        globals()[custom_flower_client_variable_name] = custom_flower_client_instance

    
'''# Assuming you have the 'selected_clients' list

# Create a list to store the Flower clients for the selected clients
flower_clients = []

# Loop through the selected clients and start the clients
for client in selected_clients:
    client_name = client['name']
    
    # Assuming you have the Flower clients stored with variable names like 'flower_client_client_1', 'flower_client_client_2', etc.
    flower_client_variable_name = f"flower_client_{client_name}"

    # Start the Flower client
    print(f"Starting Flower client for {client_name}")
    client_instance = globals()[flower_client_variable_name]
    client_instance.start()

    # Append the client to the list
    flower_clients.append(client_instance)

# Now the Flower clients are started and stored in the 'flower_clients' list
# You can access and use them as needed
'''


