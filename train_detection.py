from ultralytics import RTDETR

def train():
    # RT-DETR: Real-Time Detection Transformer
    # Escolhemos o modelo 'rtdetr-l' (large) ou 'rtdetr-x' pré-treinado
    # O arquivo .pt será baixado automaticamente na primeira vez
    model = RTDETR('rtdetr-l.pt') 
    
    print("Iniciando treinamento do Transformer de Detecção...")
    
    # Treinar
    model.train(
        data='data/detection_dataset/data.yaml', # O arquivo criado pelo passo anterior
        epochs=10,
        imgsz=640,
        batch=8,
        device=0 # Use 0 para GPU ou 'cpu'
    )
    
    print("Treino finalizado!")

if __name__ == "__main__":
    train()