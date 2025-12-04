# 🚦 Traffic Light Detection com RT-DETR (Transformer)

Este projeto implementa um sistema de detecção e classificação de semáforos em tempo real utilizando **Deep Learning** e **Computer Vision**. 

Diferente de abordagens clássicas de classificação (que apenas dizem a cor de uma imagem recortada), este projeto utiliza um **Vision Transformer (RT-DETR)** para localizar múltiplos semáforos na cena completa e identificar seus estados (Verde, Vermelho, Amarelo, Off) simultaneamente.

## 🎯 Objetivo
O projeto foi desenvolvido como parte de um estudo prático sobre **Transformers Visuais**, migrando de arquiteturas CNN tradicionais (como ResNet) para modelos de detecção de objetos de última geração (SOTA).

**Principais Funcionalidades:**
- 🧹 **Pipeline de Engenharia de Dados:** Conversão automática do dataset LISA (CSV/Anotações complexas) para o formato padrão YOLO/Detection.
- 🧠 **Treinamento com RT-DETR:** Utilização da biblioteca Ultralytics para Fine-Tuning do modelo `rtdetr-l` (Large).
- 🎥 **Inferência em Vídeo:** Script para processamento de vídeos reais com visualização de bounding boxes em tempo real.

## 🛠️ Stack Tecnológico
* **Linguagem:** Python 3.10
* **Core:** [Ultralytics](https://github.com/ultralytics/ultralytics) (RT-DETR)
* **Visão Computacional:** OpenCV
* **Manipulação de Dados:** Pandas, Scikit-Learn
* **Dataset:** LISA Traffic Light Dataset

## 📂 Estrutura do Projeto

```text
├── convert_lisa_to_detection.py  # Script de ETL: Converte anotações do LISA para .txt (YOLO format)
├── train_detection.py            # Script de configuração e treinamento do modelo
├── run_video_detection.py        # Script de inferência (aplica o modelo em vídeos)
├── requirements.txt              # Dependências do projeto
└── README.md                     # Documentação
