import torchxrayvision as xrv

# Load your model
model = xrv.models.DenseNet(weights="densenet121-res224-all")

# Print the index and the disease name
for i, disease in enumerate(model.pathologies):
    print(f"Index {i}: {disease}")