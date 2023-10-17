import client_generation as cg
CLIENT_NUM_SELECTED = 8


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
print("\nAll Clients and Losses:")
for client, loss in sorted_clients_by_loss:
    print(f"Client {client['name']} - Loss: {loss}")


