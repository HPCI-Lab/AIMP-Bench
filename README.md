# FBK Benchmarks

![uses](assets/fbk.png)

## Summary 

| Benchmark | Model Type | Resource | Why |
| - | - | - | - |
| **Compute-memory-bound** | Very deep MLP | Compute / FLOPs | Large dense layers + Small dataset |
| **I/O-bound** | Small CNN + heavy augmentation | Disk + CPU | Slow transforms + Small batches |
| **ML Scaling** | MLP | Loss, Compute, Data and Parameters | Reproduce Scaling Laws |

## Use case 1: Compute-Memory bound ML training

![Roofline](assets/roofline.png)

## Use case 2: I/O bound ML training

| GPU usage | Memory usage |
| - | - |
| ![GPU](assets/io_gpu_usage.png) | ![Memory](assets/io_memory_usage.png) |

## Use case 3: Simple ML training

![Roofline](assets/Loss.png)

## Setup

```bash
pip install -r requirements.txt
```

General folder creation

```bash
python src/preprocess/setup.py
```

Create the MNIST I/O dataset by running: 

```bash
python src/preprocess/create_MNIST_ds.py
```

## Running

```bash
python src/compute_bound.py --batch_size 64 --params 256
```

```bash
python src/io_bound.py --tform small
```

```bash
python src/ml_usecase.py --params 64 --epochs 1 --samples 4096 
```