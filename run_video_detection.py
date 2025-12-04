from ultralytics import RTDETR
import cv2

# Caminho do modelo treinado (ele cria uma pasta 'runs' automática)
# Ajuste este caminho após o treino terminar
MODEL_PATH = 'runs/detect/train/weights/best.pt' 
VIDEO_PATH = 'data/raw/video_teste.mp4'

def main():
    # Carregar modelo
    model = RTDETR(MODEL_PATH)
    
    # Abrir vídeo
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # O modelo faz a predição na imagem INTEIRA
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # O results[0].plot() já desenha as caixas e nomes na imagem
        annotated_frame = results[0].plot()
        
        cv2.imshow("DETR Traffic Light Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()