import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module): # <- almost everything in Pytorch inherits nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))
        
        # Forward method

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
         return self.weights * x + self.bias
    


if __name__ == "__main__":
    torch.manual_seed(42)

    # Create *known* parameters
    weight = 0.7
    bias = 0.3

    # Create

    start = 0
    end = 1
    step = 0.02

    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    # Create a train/test split

    train_split = int(0.8*len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    model = LinearRegressionModel()
    # Set up a loss function
    loss_fn = nn.L1Loss()

    # Set up an optimize
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.01)
    
    epochs = 200

    epoch_count = []
    loss_values = []
    test_loss_values = []

    # 0. Loop through the data
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()

        # 1. Forward pass
        y_pred = model(X_train)

        # 2. Calculate the loss

        loss = loss_fn(y_pred, y_train)

        # 3. Optimizer zero grad

        optimizer.zero_grad()

        # 4. Backpropogation

        loss.backward()

        # 5. Step the optimizer

        optimizer.step()


        ### Testing
        model.eval() # turns off different settings in the model not needed for evaluation
        with torch.inference_mode(): # turns off gradient tracking & more
            # 1. Do the forward pass
            test_pred = model(X_test)

            # Calculate the test loss
            test_loss = loss_fn(test_pred, y_test)
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")


    # Plot the loss curves
    plt.plot(epoch_count, np.array(torch.tensor(loss_values).cpu().numpy()), label = "Train Loss")
    plt.plot(epoch_count, test_loss_values, label = "Test loss")
    plt.title("Training and Testing loss curves")
    plt.ylabel("Loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()

    

    with torch.inference_mode(): # Disables all the training stuff like gradients , it makes prediction faster
        y_preds_new = model(X_test)

    print(y_test)
    print("--------------")
    print(y_preds_new)