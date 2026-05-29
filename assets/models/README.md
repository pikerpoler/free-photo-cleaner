Place `mobilenet_v3_small.onnx` in this directory.

To generate the ONNX model:

```python
import torch
import torchvision.models as models

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
# Remove classifier, keep only features (produces 576-dim embeddings)
model.classifier = torch.nn.Identity()
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, "mobilenet_v3_small.onnx",
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    opset_version=13,
)
```

The resulting file should be ~7MB and produce 576-dimensional embeddings.
