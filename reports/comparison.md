| Modelo | Train | Accuracy | Macro-F1 | F1 neutro | Err. polaridad | Latencia p50 (1 texto) | Throughput (lotes de 32) | Tamaño |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + LogReg (30k) | 30k | 0.7214 | **0.6871** | 0.4738 | 4.09% | 1.5 ms | 3691 textos/s | 4 MB |
| TF-IDF + LogReg (completo) | 197k | 0.7413 | **0.7080** | 0.4939 | 2.63% | 1.2 ms | 3977 textos/s | 11 MB |
| DistilBETO (30k, 2 épocas, sin pesos de clase) | 30k | 0.7703 | **0.7085** | 0.4428 | 3.15% | 7.7 ms | 105 textos/s | 257 MB |
| DistilBETO (PyTorch) | 100k | 0.7589 | **0.7266** | 0.5129 | 1.46% | 10.0 ms | 78 textos/s | 257 MB |
| DistilBETO (ONNX fp32) | 100k | 0.7589 | **0.7266** | 0.5129 | 1.46% | 4.7 ms | 96 textos/s | 258 MB |
| DistilBETO (ONNX int8) | 100k | 0.7655 | **0.7347** | 0.5265 | 1.24% | 5.8 ms | 88 textos/s | 64 MB |
