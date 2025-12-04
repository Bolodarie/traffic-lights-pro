import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Importações dos nossos módulos
from src.config import DEFAULT_IMG_SIZE, PROCESSED_DATA_DIR, DATA_DIR
from src.data.dataset import TrafficLightDataset
from src.data.transforms import get_train_transforms, get_valid_transforms
from src.models.factory import get_model
from src.models.hybrid_model import CNNTransformerHybrid
from src.engine.trainer import train_one_epoch, validate
from src.engine.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="hybrid", help="resnet18, efficientnet ou hybrid")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_csv", type=str, default="data/dataset.csv", help="Caminho para o CSV de dados")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # --- 1. PREPARAÇÃO DOS DADOS (SIMULAÇÃO) ---
    # Para testar, vamos criar um CSV falso se ele não existir
    if not os.path.exists(args.data_csv):
        print(f"AVISO: {args.data_csv} não encontrado. Criando dummy para teste.")
        # Cria dataframe dummy (você deve substituir isso pela leitura real do LISA)
        # Assumindo que você tenha imagens em data/raw/test_images
        dummy_data = {
            'filename': ['img1.jpg', 'img2.jpg'] * 10, # Nomes ficticios
            'label': [0, 1] * 10
        }
        # Nota: Isso vai falhar se as imagens não existirem fisicamente.
        # Certifique-se de ter imagens na pasta apontada pelo dataset.
    
    # Em um cenário real, você leria o CSV real aqui
    # df = pd.read_csv(args.data_csv)
    # train_df, val_df = train_test_split(df, test_size=0.2)
    
    # MOCKUP PARA O EXEMPLO RODAR (Substitua pela lógica acima)
    print("!!! ATENÇÃO: Carregando Dataset Mockado (Sem dados reais, vai dar erro se não tiver imagens) !!!")
    # Aqui você deve passar o dataframe real
    # train_dataset = TrafficLightDataset(train_df, root_dir='data/raw', transform=get_train_transforms())
    # val_dataset = TrafficLightDataset(val_df, root_dir='data/raw', transform=get_valid_transforms())

    # --- 2. DATALOADERS ---
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("Pulei o carregamento de dados real para evitar erro de arquivo inexistente.")
    print("Para funcionar: coloque imagens em data/raw e aponte um CSV válido.")
    return # Remova isso quando configurar os dados

    # --- 3. MODELO ---
    if args.model_type == 'hybrid':
        print("Inicializando Modelo Híbrido (CNN + Transformer)...")
        model = CNNTransformerHybrid(num_classes=4).to(device)
    else:
        print(f"Inicializando Modelo TIMM: {args.model_type}...")
        model = get_model(args.model_type, num_classes=4).to(device)

    # --- 4. OTIMIZADOR E LOSS ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    evaluator = Evaluator(device, num_classes=4)

    # --- 5. LOOP DE TREINO ---
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, evaluator, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val F1: {val_metrics['MulticlassF1Score']:.4f}")
        print(f"Val Acc: {val_metrics['MulticlassAccuracy']:.4f}")

        # Salvar melhor modelo
        torch.save(model.state_dict(), f"outputs/model_epoch_{epoch}.pth")

if __name__ == "__main__":
    import os
    main()