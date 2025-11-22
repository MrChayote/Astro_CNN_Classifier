import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = '/home/chayote/SpaceNet.FLARE.imam_alam'
BATCH_SIZE = 64  # Con B0 podrías subir a 128 sin problemas en tu 5070 Ti
NUM_WORKERS = 16 
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

print(f"Entrenando EfficientNet_B0 en: {DEVICE}")

# --- 1. CONSTRUCCIÓN DEL MODELO ---
def build_efficientnet(num_classes):
    print("Cargando EfficientNet_B0...")
    
    # Cargamos los pesos y sus transformaciones específicas
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    # Obtenemos el preprocesamiento automático que recomienda el creador del modelo
    auto_transforms = weights.transforms()
    
    # Congelar capas base (Feature Extractor)
    for param in model.parameters():
        param.requires_grad = False
        
    # --- LA PARTE CRÍTICA: CAMBIAR EL CLASIFICADOR ---
    # EfficientNet tiene un bloque 'classifier' que es: Dropout -> Linear
    # Queremos reemplazar solo la parte Linear (índice 1)
    
    num_ftrs = model.classifier[1].in_features
    
    # Reconstruimos el clasificador final
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model.to(DEVICE), auto_transforms

# --- 2. PREPARACIÓN DE DATOS (Usando Auto-Transforms) ---
def get_dataloaders(data_dir, model_transforms):
    # Aplicamos Data Augmentation adicional sobre las transformaciones base del modelo
    # EfficientNet es sensible, así que usaremos sus bases + nuestras rotaciones
    
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        model_transforms, # Aquí van el Resize y Normalize oficiales de EfficientNet
    ])

    # Dataset completo
    full_dataset = datasets.ImageFolder(data_dir) # No aplicamos transform aquí todavía
    
    # Dividir
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Asignar transformaciones
    # Truco sucio pero efectivo para asignar transforms distintos después del split
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)

    train_set = TransformedDataset(train_dataset, transform=train_transforms)
    val_set = TransformedDataset(val_dataset, transform=model_transforms) # Solo resize/norm

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader, full_dataset.classes
    
# --- 3. BUCLE DE ENTRENAMIENTO (Reutilizable) ---
# (Es idéntico al de ResNet, lo pongo resumido para que corra directo)
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        print(f'\nÉpoca {epoch+1}/{epochs}')
        
        # Train
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, leave=True)
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_description(f"Train")
            loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_acc += torch.sum(preds == labels.data)
        
        acc = val_acc.double() / len(val_loader.dataset)
        print(f'Validation Accuracy: {acc:.4f}')
        
    return model

# --- EJECUCIÓN ---
if __name__ == '__main__':
    # 1. Construir modelo primero para obtener sus transformaciones ideales
    # (Necesitamos saber cuántas clases hay, usaremos 6 por defecto o leemos carpeta antes)
    # Truco: Leemos las clases rápido primero
    temp_dataset = datasets.ImageFolder(DATA_PATH)
    class_names = temp_dataset.classes
    print(f"Clases: {class_names}")
    
    model, auto_transforms = build_efficientnet(len(class_names))
    
    # 2. Cargar datos con las transforms de EfficientNet
    train_loader, val_loader, _ = get_dataloaders(DATA_PATH, auto_transforms)
    
    # 3. Optimizador
    criterion = nn.CrossEntropyLoss()
    # Optimizamos 'model.classifier' en lugar de 'model.fc'
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    
    # 4. Entrenar
    model_ft = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=NUM_EPOCHS)
    
    # 5. Guardar
    torch.save(model_ft.state_dict(), 'efficientnet_b0_astronomia.pth')
    print("¡EfficientNet guardado!")