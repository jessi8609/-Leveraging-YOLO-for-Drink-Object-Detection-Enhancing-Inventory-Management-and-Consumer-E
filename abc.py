from ultralytics import YOLO
import os


def verify_dataset():
    """
    Verify the dataset directory structure.
    """
    train_images = r"C:\Users\13\Desktop\drink\train\images"
    train_labels = r"C:\Users\13\Desktop\drink\train\labels"
    val_images = r"C:\Users\13\Desktop\drink\valid\images"
    val_labels = r"C:\Users\13\Desktop\\drink\valid\labels"

    # Check if the directories exist
    if not os.path.exists(train_images):
        raise FileNotFoundError(f"Train images directory not found: {train_images}")
    if not os.path.exists(train_labels):
        raise FileNotFoundError(f"Train labels directory not found: {train_labels}")
    if not os.path.exists(val_images):
        raise FileNotFoundError(f"Validation images directory not found: {val_images}")
    if not os.path.exists(val_labels):
        raise FileNotFoundError(f"Validation labels directory not found: {val_labels}")

    print("Dataset verified successfully.")


# def create_data_yaml():
#     """
#     Create the 'data.yaml' configuration file for training.
#     """
#     yaml_content = """train: dataset/train/images
# val: dataset/val/images
# nc: 1
# names: ['class_name']
# """
#     # Save the content to a data.yaml file
#     with open("dataset/data.yaml", "w") as yaml_file:
#         yaml_file.write(yaml_content)
#     print("data.yaml file created successfully.")


def train_model():
    """
    Train the YOLO model.
    """
    verify_dataset()  # Ensure the dataset structure is correct
    # create_data_yaml()  # Create data.yaml configuration file

    # Load YOLO model
    model = YOLO("yolo11n.pt")  # Replace with your YOLO model path if necessary

    # Train the model
    results = model.train(
        data=r"C:\Users\13\Desktop\drink\data.yaml",  # Path to YAML file
        epochs=100,                # Number of epochs
        imgsz=640,                 # Image size
        batch=16,                  # Batch size
        workers=8,                 # Number of workers for data loading
        device=0,                  # Use GPU (0) or CPU ('cpu')
        pretrained=True,           # Use pretrained weights
        patience=50,               # Early stopping patience
        save=True,                 # Save weights
        save_period=10,            # Save every 10 epochs
        single_cls=True            # Train on a single class
    )

    print("Training completed successfully.")


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred: {e}")
 # type: ignore