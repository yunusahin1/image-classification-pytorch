import torch
from training import training_loop
from utils import config_yaml
import matplotlib.pyplot as plt
import os

def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = config_yaml()
    
    training_config = config.get("training", {})
    epochs = training_config.get("epochs", 10)
    learning_rate = training_config.get("learning_rate", 0.001)
    batch_size = training_config.get("batch_size", 32)
    
    print("Starting model training...")
    training_results = training_loop(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
    
    model = training_results['model']
    train_losses = training_results['train_losses']
    test_losses = training_results['test_losses']
    train_accuracies = training_results['train_accuracies']
    test_accuracies = training_results['test_accuracies']
    label_encoder = training_results['label_encoder']
    
    print("\nTraining completed!")
    print(f"Final train accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final test accuracy: {test_accuracies[-1]:.4f}")
    
    if config.get("save_model", False):
        model_dir = config.get("model_dir", "./models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "image_classifier.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_encoder': label_encoder,
        }, model_path)
        print(f"Model saved to {model_path}")
    
    if config.get("plot_training", False):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.tight_layout()
        
        if config.get("save_plots", False):
            plots_dir = config.get("plots_dir", "./plots")
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, "training_metrics.png"))
            print(f"Training plots saved to {plots_dir}")
        else:
            plt.show()

if __name__ == "__main__":
    main()
