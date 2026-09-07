import time
import torch
import torch.nn as nn
from torchvision import models

def count_parameters(model):
 return sum(p.numel() for p in model.parameters()) / 1e6

def benchmark_model(name, model_fn, num_classes=5, num_warmup=10, num_runs=50):
 try:
 model = model_fn()
 model.eval()

 # CPU Latency benchmark
 device = torch.device("cpu")
 model.to(device)
 dummy_input = torch.randn(1, 3, 224, 224).to(device)

 # Warmup
 with torch.no_grad():
 for _ in range(num_warmup):
 _ = model(dummy_input)

 # Benchmark
 start_time = time.perf_counter()
 with torch.no_grad():
 for _ in range(num_runs):
 _ = model(dummy_input)
 end_time = time.perf_counter()

 avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
 fps = 1000 / avg_latency_ms
 params_m = count_parameters(model)

 print(f"| {name:<22} | {params_m:6.2f}M | {avg_latency_ms:7.2f} ms | {fps:6.1f} FPS |")
 return {
 'name': name,
 'params_m': params_m,
 'latency_ms': avg_latency_ms,
 'fps': fps
 }
 except Exception as e:
 print(f"Error benchmarking {name}: {e}")
 return None

if __name__ == "__main__":
 print("=== PyTorch Lightweight Backbone CPU Benchmark for Drone Edge Devices ===")
 print(f"PyTorch Version: {torch.__version__}")
 print("-" * 65)
 print(f"| {'Backbone Model':<22} | {'Params':<7} | {'CPU Latency':<11} | {'Throughput':<10} |")
 print("-" * 65)

 models_to_test = [
 ("MobileNetV3-Small", lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)),
 ("MobileNetV3-Large", lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)),
 ("EfficientNet-B0", lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)),
 ("ShuffleNetV2 (1.0x)", lambda: models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)),
 ("ResNet18 (Baseline)", lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT)),
 ("ResNet34", lambda: models.resnet34(weights=models.ResNet34_Weights.DEFAULT)),
 ]

 results = []
 for name, model_fn in models_to_test:
 res = benchmark_model(name, model_fn)
 if res:
 results.append(res)

 print("-" * 65)
