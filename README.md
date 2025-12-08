# CBIR Streamlit App  
Autores: Alvaro Huisman, Esteban Moreno Mendoza

Buscador de imágenes por contenido con varios extractores (color, SIFT/ORB BoW, ResNet50, EfficientNet-B0) y FAISS. El repo incluye todo precomputado para ejecutar la app sin recalcular índices.

## Estructura entregada
```
cbir/
├─ app.py
├─ requirements.txt
├─ database/
│  ├─ db_train.csv
│  ├─ db_test.csv
│  └─ db.csv
├─ dataset/
│  ├─ headset/train|test/*.jpg
│  ├─ keyboard/train|test/*.jpg
│  ├─ mouse/train|test/*.jpg
│  ├─ speakers/train|test/*.jpg
│  └─ webcam/train|test/*.jpg
├─ faiss_indexes/                # índices + metadata ya construidos
│  ├─ color_histogram_db_train_l2.index / .csv
│  ├─ sift_bow_db_train_l2.index / .csv
│  ├─ orb_bow_db_train_l2.index  / .csv
│  ├─ resnet50_db_train_l2.index / .csv
│  └─ efficientnet_b0_db_train_l2.index / .csv
├─ features/
│  ├─ sift_codebook_k512.npy
│  └─ orb_codebook_k512.npy
├─ src/
│  ├─ extractors/ (color_histogram.py, sift_bow.py, orb_bow.py, resnet.py, efficientnet.py, ...)
│  ├─ build_features.py, build_faiss_index.py
│  ├─ build_sift_codebook.py, build_orb_codebook.py
│  └─ evaluate_retrieval.py
└─ programas_auxiliares/
   ├─ database_creator.py
   └─ restructure_dataset.py
```

## Instalación y ejecución (usuario final)
1) Descargar las imágenes: baja la carpeta `dataset/` desde **[Google Drive](https://drive.google.com/drive/folders/1KLPWNsyXwuI1PJ55Nbxz8DWCcqx3BcZS?usp=drive_link)** y colócala en la raíz del proyecto, manteniendo `dataset/<clase>/train|test/*.jpg`.
2) Crear entorno y dependencias:
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
3) Ejecutar la app:
```bash
streamlit run app.py
```
No hace falta recalcular índices ni codebooks.

## Datasets e índices precomputados
- Imágenes: solo están en Google Drive (**[Google Drive](https://drive.google.com/drive/folders/1KLPWNsyXwuI1PJ55Nbxz8DWCcqx3BcZS?usp=drive_link)**). Descarga `dataset/` y ponla en la raíz.
- Índices/codebooks: ya versionados en `faiss_indexes/` y `features/` (no recalcular).
- Código en GitHub: **[Repositorio](https://github.com/alvaro-huisman/practica_cbir)**.

## Métricas obtenidas (db_train → db_test, k = 1/5/10)
- Color histogram: mAP 0.4364 | hit@1 0.33 | hit@5 0.71 | hit@10 0.88
- SIFT BoW: mAP 0.5544 | hit@1 0.45 | hit@5 0.84 | hit@10 0.96
- ORB BoW: mAP 0.5355 | hit@1 0.45 | hit@5 0.84 | hit@10 0.95
- ResNet50: mAP 0.9587 | hit@1 0.97 | hit@5 1.00 | hit@10 1.00
- EfficientNet-B0: mAP 0.9716 | hit@1 0.95 | hit@5 1.00 | hit@10 1.00

## Si cambias imágenes (opcional)
Solo si quieres rehacer índices: usa `programas_auxiliares/restructure_dataset.py`, luego `src/build_*` para codebooks, features e índices, y `src/evaluate_retrieval.py` para recomputar métricas.
