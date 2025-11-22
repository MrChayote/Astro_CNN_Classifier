import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
import os

# --- 1. CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = '/home/chayote/SpaceNet.FLARE.imam_alam'
# ViT es pesado en VRAM. Con tu 5070 Ti, 32 o 64 deberían ir bien. 
# Si te sale "Out of Memory", baja a 32.
BATCH_SIZE = 64 
NUM_WORKERS = 16
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5 # <--- OJO: Los Transformers prefieren Learning Rates más bajos (0.00002)

print(f"Preparando Vision Transformer en: {DEVICE}")

# --- 2. PREPARACIÓN DE DATOS ---
# Usamos el procesador oficial de ViT para asegurar que la normalización sea exacta
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Definimos transformaciones estándar usando los valores del procesador
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ]),
}

def get_dataloaders():
    full_dataset = datasets.ImageFolder(DATA_PATH, transform=data_transforms['train'])
    class_names = full_dataset.classes
    print(f"Clases detectadas: {class_names}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Asignar transform correcta a validación
    val_dataset.dataset.transform = data_transforms['val']

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader, class_names

# --- 3. CONSTRUIR EL MODELO ViT ---
def build_vit_model(num_classes):
    print("Descargando modelo ViT de Hugging Face...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True # Necesario para ajustar la capa final a 6 clases
    )
    return model.to(DEVICE)

# --- 4. BUCLE DE ENTRENAMIENTO (Adaptado para Hugging Face) ---
def train_vit(model, train_loader, val_loader, optimizer, epochs=10):
    for epoch in range(epochs):
        print(f'\nÉpoca {epoch+1}/{epochs}')
        
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        loop = tqdm(train_loader, leave=True)
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Hugging Face devuelve un objeto, no un tensor directo
            outputs = model(inputs) 
            logits = outputs.logits # <--- AQUÍ ESTÁ LA CLAVE
            
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Estadísticas
            _, preds = torch.max(logits, 1)
            train_loss += loss.item() * inputs.size(0)
            train_correct += torch.sum(preds == labels.data)
            total_train += labels.size(0)
            
            loop.set_description(f"Train")
            loop.set_postfix(loss=loss.item())
            
        epoch_acc = train_correct.double() / total_train
        epoch_loss = train_loss / total_train
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                logits = outputs.logits
                _, preds = torch.max(logits, 1)
                val_correct += torch.sum(preds == labels.data)
                total_val += labels.size(0)
        
        val_acc = val_correct.double() / total_val
        print(f"Validation Acc: {val_acc:.4f}")

    return model

# --- 5. EJECUCIÓN ---
if __name__ == '__main__':
    train_loader, val_loader, class_names = get_dataloaders()
    
    model = build_vit_model(len(class_names))
    
    # Usamos AdamW (Mejor para Transformers) y un LR bajo
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    model_ft = train_vit(model, train_loader, val_loader, optimizer, epochs=NUM_EPOCHS)
    
    # Guardar modelo (Formato Hugging Face)
    model_ft.save_pretrained('./vit_astronomia_model')
    print("¡Modelo ViT guardado!")