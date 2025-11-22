import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm  # Barra de progreso bonita

# --- CONFIGURACIÓN DE HARDWARE Y RUTAS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = '/home/chayote/SpaceNet.FLARE.imam_alam'
BATCH_SIZE = 64          # Tu 5070 Ti aguanta esto y más. Si te sobra VRAM, sube a 128.
NUM_WORKERS = 16         # ¡Aquí está la magia! Usamos 16 hilos de tu Ryzen 9900X.
NUM_EPOCHS = 10          # Empezamos con 10 para probar rápido
LEARNING_RATE = 0.001

print(f"Usando dispositivo: {DEVICE}")
print(f"Ruta de datos: {DATA_PATH}")

# --- 1. TRANSFORMACIONES (Data Augmentation) ---
# ResNet-50 espera imágenes de 224x224.
# Normalizamos con los valores estándar de ImageNet.

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),        # Rotación para invarianza (importante en astro)
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),      # Arriba/Abajo es igual en el espacio
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 2. CARGA DE DATOS OPTIMIZADA ---
def get_dataloaders(data_dir):
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    
    # Mapeo de clases para confirmar
    print(f"Clases encontradas: {full_dataset.classes}")
    
    # Dividir 80% Entrenamiento / 20% Validación
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # IMPORTANTE: Aplicar la transformación de validación al set de validación
    # (Por defecto hereda la de train con rotaciones, lo cual no queremos para validar)
    val_dataset.dataset.transform = data_transforms['val']

    # DataLoaders con optimización para Ryzen/RTX
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, # Carga paralela en CPU
        pin_memory=True          # Acelera paso RAM -> VRAM
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.classes

# --- 3. DEFINICIÓN DEL MODELO ---
def build_resnet50(num_classes):
    print("Descargando/Cargando ResNet-50 pre-entrenada...")
    # 'weights="DEFAULT"' usa los mejores pesos disponibles de ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Congelar los pesos base (Opcional: Si tienes muchas imágenes, puedes descongelar)
    # Por ahora congelamos para que sea rápido y solo aprenda la clasificación final.
    for param in model.parameters():
        param.requires_grad = False
    
    # Reemplazar la capa final (Fully Connected)
    # ResNet50 tiene 2048 entradas en su última capa.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model.to(DEVICE)

# --- 4. BUCLE DE ENTRENAMIENTO UNIVERSAL ---
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        print(f'\nÉpoca {epoch+1}/{epochs}')
        print('-' * 10)

        # --- FASE DE ENTRENAMIENTO ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Barra de progreso
        loop = tqdm(train_loader, leave=True)
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad() # Limpiar gradientes
            
            outputs = model(inputs) # Predicción
            loss = criterion(outputs, labels) # Calcular error
            
            loss.backward() # Backpropagation
            optimizer.step() # Actualizar pesos

            # Estadísticas
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Actualizar barra
            loop.set_description(f"Entrenando")
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # --- FASE DE VALIDACIÓN ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad(): # No calcular gradientes para validar (ahorra VRAM)
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

    return model

# --- 5. EJECUCIÓN PRINCIPAL ---
if __name__ == '__main__':
    # 1. Preparar datos
    train_loader, val_loader, class_names = get_dataloaders(DATA_PATH)
    
    # 2. Construir Modelo
    model = build_resnet50(len(class_names))
    
    # 3. Configurar optimizador y función de pérdida
    criterion = nn.CrossEntropyLoss()
    # Optimizamos solo la última capa (fc) porque el resto está congelado
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    # 4. ¡Entrenar!
    model_ft = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=NUM_EPOCHS)
    
    # 5. Guardar modelo
    torch.save(model_ft.state_dict(), 'resnet50_astronomia.pth')
    print("Modelo guardado exitosamente.")