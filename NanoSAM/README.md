NanoSAM-ONNX · Evaluación con prompts y métricas

Este cuaderno evalúa NanoSAM exportado a ONNX (encoder y decoder con ONNX Runtime) sobre un conjunto de datos de PCB, midiendo IoU, Dice, Precisión, Recall y tiempo a 1024×1024 con diferentes prompts. Está pensado para Google Colab con versiones fijas (NumPy 1.26.4, OpenCV 4.8.1.78, ONNX Runtime 1.17.3; opcionalmente onnxruntime-gpu 1.17.3). Los modelos resnet18_image_encoder.onnx y mobile_sam_mask_decoder.onnx deben estar en /content/drive/MyDrive/Colab Notebooks/TFG/NanoSam/, y las imágenes + anotaciones .json en /content/drive/MyDrive/Colab Notebooks/SolDef_AI/Labeled/; a partir de los .json se rasterizan máscaras binarias en generated_masks/. El flujo monta Drive, clona el repo oficial de NanoSAM, limpia caches e instala dependencias; luego carga los ONNX y verifica entradas y salidas. Cada imagen se preprocesa, se normaliza tipo ImageNet y se convierte a NCHW float32; el encoder produce embeddings (1,256,64,64) que, junto a coordenadas de prompts (positivos=1) y sin máscara previa, se pasan al decoder para obtener cuatro hipótesis (logits 256×256) y sus iou_predictions. Se elige la hipótesis con mayor IoU predicho, se binariza (logit>0 ≡ p>0.5), se reescala a 1024 y opcionalmente al tamaño original para visualización. Se evalúan cuatro modalidades de prompt (point, box, box+point, multipoint) y un barrido por número de puntos K, computando métricas en 1024 y registrando tiempos; se guardan visualizaciones.

Resultados obtenidos:

| K\_points |   N | IoU\_mean | IoU\_std | Dice\_mean | Dice\_std | Prec\_mean | Prec\_std | Rec\_mean | Rec\_std | Time\_mean\_s | FPS\_mean |
| --------: | --: | --------: | -------: | ---------: | --------: | ---------: | --------: | --------: | -------: | ------------: | --------: |
|         1 | 428 |    0.5132 |   0.2953 |     0.6293 |    0.2554 |     0.7566 |    0.2790 |    0.7199 |   0.3338 |        0.0946 |   10.5748 |
|         2 | 428 |    0.5886 |   0.2757 |     0.7028 |    0.2226 |     0.7237 |    0.2861 |    0.8412 |   0.2473 |        0.0955 |   10.4748 |
|         3 | 428 |    0.6434 |   0.2651 |     0.7502 |    0.2048 |     0.7453 |    0.2764 |    0.8777 |   0.1982 |        0.0954 |   10.4842 |
|         5 | 428 |    0.6587 |   0.2511 |     0.7658 |    0.1890 |     0.7336 |    0.2761 |    0.9071 |   0.1463 |        0.0965 |   10.3634 |
|        10 | 428 |    0.7043 |   0.2523 |     0.7992 |    0.1851 |     0.7372 |    0.2690 |    0.9540 |   0.0718 |        0.1000 |    9.9966 |





| Método     |   N | IoU\_mean | IoU\_std | Dice\_mean | Dice\_std | Prec\_mean | Prec\_std | Rec\_mean | Rec\_std | Time\_mean\_s | FPS\_mean |
| :--------- | --: | --------: | -------: | ---------: | --------: | ---------: | --------: | --------: | -------: | ------------: | --------: |
| Caja       | 428 |    0.2562 |   0.1701 |     0.3804 |    0.2058 |     0.2578 |    0.1710 |    0.9177 |   0.2373 |        0.0997 |     10.03 |
| Caja+Punto | 428 |    0.2562 |   0.1701 |     0.3804 |    0.2058 |     0.2578 |    0.1710 |    0.9177 |   0.2373 |        0.0993 |     10.07 |
| Multipunto | 428 |    0.6689 |   0.2568 |     0.7721 |    0.1927 |     0.7318 |    0.2757 |    0.9186 |   0.1378 |        0.0991 |     10.09 |
| Punto      | 428 |    0.2912 |   0.3193 |     0.3671 |    0.3456 |     0.6132 |    0.4277 |    0.4142 |   0.4254 |        0.0966 |     10.36 |




