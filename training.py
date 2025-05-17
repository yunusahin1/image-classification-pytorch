from cnn import CNN
import torch
from torch import nn
from image_preprocessor import label_images, image_to_tensors, create_train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader


def training_loop(epochs: int, learning_rate: float, batch_size: int):
    labeled_df = label_images()
    images, labels = image_to_tensors(labeled_df)
    
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    
    data = create_train_test_split(images, numeric_labels)
    train_dataset, test_dataset = data['train_dataset'], data['test_dataset']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        cnn.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_images, batch_labels in train_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = cnn(batch_images)
            loss = loss_fn(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        cnn.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_images, batch_labels in test_loader:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                
                outputs = cnn(batch_images)
                loss = loss_fn(outputs, batch_labels)
                
                running_loss += loss.item() * batch_images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_acc)
        
        print(f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}")
        print(f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}")
    
    return {
        'model': cnn,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'label_encoder': label_encoder
    }

