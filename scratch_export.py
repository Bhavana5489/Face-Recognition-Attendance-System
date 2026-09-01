import sys
sys.path.append("d:/projects/face_detection/anti_spoofing/src")
from model_lib.MiniFASNet import MiniFASNetV1SE, MiniFASNetV2
import torch
import os

os.makedirs("d:/projects/face_detection/gazepass/models/liveness", exist_ok=True)

# Export V1SE
model_v1 = MiniFASNetV1SE(conv6_kernel=(5,5))
state_dict = torch.load("d:/projects/face_detection/anti_spoofing/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth", map_location='cpu')
new_state_dict = {}
for k, v in state_dict.items():
    k = k.replace('module.', '')
    k = k.replace('se_fc', 'se_module.fc')
    k = k.replace('se_bn', 'se_module.bn')
    new_state_dict[k] = v
model_v1.load_state_dict(new_state_dict)
model_v1.eval()
dummy_input = torch.randn(1, 3, 80, 80)
torch.onnx.export(model_v1, dummy_input, "d:/projects/face_detection/gazepass/models/liveness/minifasnet_v1se.onnx", 
                  input_names=['input'], output_names=['output'], opset_version=11)
print("Exported V1SE")

# Export V2
model_v2 = MiniFASNetV2(conv6_kernel=(5,5))
state_dict = torch.load("d:/projects/face_detection/anti_spoofing/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth", map_location='cpu')
new_state_dict = {}
for k, v in state_dict.items():
    k = k.replace('module.', '')
    k = k.replace('se_fc', 'se_module.fc')
    k = k.replace('se_bn', 'se_module.bn')
    new_state_dict[k] = v
model_v2.load_state_dict(new_state_dict)
model_v2.eval()
dummy_input = torch.randn(1, 3, 80, 80)
torch.onnx.export(model_v2, dummy_input, "d:/projects/face_detection/gazepass/models/liveness/minifasnet_v2.onnx", 
                  input_names=['input'], output_names=['output'], opset_version=11)
print("Exported V2")
