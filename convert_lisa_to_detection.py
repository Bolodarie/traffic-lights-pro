import pandas as pd
import os
import shutil
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2

# --- CONFIGURAÇÃO ---
LISA_ROOT = "data/archive" 
OUTPUT_DIR = "data/detection_dataset"

# Mapeamento de Classes
CLASS_MAP = {
    'go': 0, 'goLeft': 0, 'goForward': 0,          # 0 = Green
    'stop': 1, 'stopLeft': 1,                      # 1 = Red
    'warning': 2, 'warningLeft': 2,                # 2 = Yellow
}

def convert_to_yolo_format(row, img_width, img_height):
    x1, y1, x2, y2 = row['Upper left corner X'], row['Upper left corner Y'], row['Lower right corner X'], row['Lower right corner Y']
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + (w / 2)
    y_center = y1 + (h / 2)
    return [
        x_center / img_width,
        y_center / img_height,
        w / img_width,
        h / img_height
    ]

def map_all_images(root_dir):
    """
    Cria um dicionário {nome_do_arquivo: caminho_completo}
    varrendo todas as subpastas.
    """
    print("Mapeando todas as imagens na pasta (pode levar alguns segundos)...")
    image_map = {}
    # Procura recursiva por jpg e png
    patterns = [os.path.join(root_dir, "**", "*.jpg"), os.path.join(root_dir, "**", "*.png")]
    
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            filename = os.path.basename(path)
            # Se houver duplicatas, o último encontrado prevalece (geralmente ok no LISA)
            image_map[filename] = path
            
    print(f"Mapeamento concluído! {len(image_map)} imagens encontradas no disco.")
    return image_map

def main():
    # 1. Limpar execução anterior para garantir
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    # 2. Encontrar CSVs
    search_path = os.path.join(LISA_ROOT, "**", "*.csv")
    csv_files = glob.glob(search_path, recursive=True)
    
    if not csv_files:
        print(f"ERRO: Nenhum CSV encontrado em {LISA_ROOT}")
        return

    print(f"Lendo {len(csv_files)} arquivos CSV...")
    df_list = []
    for f in csv_files:
        try:
            temp_df = pd.read_csv(f, sep=None, engine='python')
            df_list.append(temp_df)
        except:
            pass

    if not df_list:
        print("Erro: Não foi possível ler nenhum CSV.")
        return

    df = pd.concat(df_list, ignore_index=True)
    
    # 3. Mapear Imagens Reais no Disco
    image_map = map_all_images(LISA_ROOT)
    
    # Criar pastas de saída
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

    # Filtrar apenas imagens que realmente temos no disco e anotações válidas
    df['basename'] = df['Filename'].apply(os.path.basename)
    df = df[df['basename'].isin(image_map.keys())]
    df = df[df['Annotation tag'].isin(CLASS_MAP.keys())]
    
    if len(df) == 0:
        print("ERRO CRÍTICO: Nenhuma correspondência entre CSV e Imagens encontradas.")
        print("Verifique se você baixou as imagens corretamente.")
        return

    unique_images = df['basename'].unique()
    train_imgs, val_imgs = train_test_split(unique_images, test_size=0.2, random_state=42)
    
    print(f"Processando {len(unique_images)} imagens válidas...")
    
    count = 0
    for img_name in tqdm(unique_images):
        split = 'train' if img_name in train_imgs else 'val'
        
        real_path = image_map[img_name]
        
        # Copiar imagem
        new_filename = f"{count}_{img_name}" # Prefixo para evitar conflito
        dest_path = f"{OUTPUT_DIR}/{split}/images/{new_filename}"
        shutil.copy(real_path, dest_path)
        
        # Pegar dimensões para normalizar YOLO
        img = cv2.imread(real_path)
        if img is None: continue
        h_img, w_img, _ = img.shape

        # Criar Labels
        img_annotations = df[df['basename'] == img_name]
        label_path = f"{OUTPUT_DIR}/{split}/labels/{new_filename.replace('.jpg', '.txt').replace('.png', '.txt')}"
        
        with open(label_path, 'w') as f:
            for _, row in img_annotations.iterrows():
                tag = row['Annotation tag']
                cls = CLASS_MAP[tag]
                bbox = convert_to_yolo_format(row, w_img, h_img)
                f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        count += 1

    # Criar YAML
    yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)}
train: train/images
val: val/images

names:
  0: Green
  1: Red
  2: Yellow
"""
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"SUCESSO! {count} imagens preparadas.")
    print("Agora pode rodar o 'python train_detection.py'")

if __name__ == "__main__":
    main()