import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

from numpy import random


random_generator = random.default_rng(501)


def train_model_one_boot(model, sample_loader, criterion, optimizer, epochs=5, device=None):
    model.train()
    for ep in range(epochs):
        train_loss = 0
        for images, labels in sample_loader:
            images, labels = images.to(device), labels.to(device)
            predictions, _ = model(images)
            
            loss = criterion(predictions, labels)
            train_loss += loss.item() * images.size(0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_loss = train_loss / len(sample_loader.dataset)
        print("Epoch: " + str(ep + 1) + " Loss: " + str(epoch_loss))
    return model, optimizer, epoch_loss


def compute_preds_test(model, test_data, criterion, dist, device=None):
    model.eval()
    valid_loss = 0
    correctOut = 0.0
    total = 0
    likelihood_est = 0.0
    
    for images, labels in test_data:
        images, labels = images.to(device), labels.to(device)
        predictions, probs = model(images)
        
        dist = np.append(dist, probs.cpu().detach().numpy())
        
        _, outClass = predictions.max(1)
        
        loss = criterion(predictions, labels)
        
        prednp = outClass.cpu().detach().numpy()
        labelnp = labels.cpu().detach().numpy()
        
        likelihood_est += -np.sum(labelnp * np.log(prednp + 1e-9))
        
        valid_loss += loss.item() * images.size(0)
        
        total += labels.size(0)
        correctOut += (outClass == labels).sum()
    
    loss = valid_loss / len(test_data.dataset)
    likelihood_est /= len(test_data.dataset)
    accuracy = (float)(correctOut / total)
    print("Loss in test: " + str(loss))
    print("Accuracy in test: {:.2f}%".format(100 * accuracy))
    print("-------------------")
    return model, loss, 100 * accuracy, dist, likelihood_est


def get_sample_data(train_dataset: Dataset, repetitions, x=200, batch_size=16):
    sample_indices = random_generator.integers(low=0, high=len(train_dataset), size=x)
    
    for ind in sample_indices:
        repetitions[str(ind)] += 1
    
    train_subset = Subset(train_dataset, sample_indices)
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
    
    return train_loader, repetitions
