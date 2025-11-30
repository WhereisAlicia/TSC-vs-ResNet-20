import torchtime.models as models

model = models.InceptionTime(in_channels=1, num_classes=10)
print(isinstance(model, torch.nn.Module))
print(hasattr(model, 'eval'))
print(hasattr(model, 'train'))
