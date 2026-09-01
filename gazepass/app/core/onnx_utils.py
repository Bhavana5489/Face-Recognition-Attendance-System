import cv2

def optimize_onnx_net(net):
    """
    Applies Phase 16 deployment optimizations.
    Configures the ONNX network to use CUDA if available,
    otherwise falls back to highly optimized CPU backend (OpenVINO/MKL).
    """
    # Try to set CUDA backend and target
    try:
        # If cv2 was compiled with CUDA support, this will enable GPU acceleration
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # Dummy forward pass to trigger backend initialization/compilation
        # but we don't do it here since input size isn't known yet.
    except Exception:
        # Fallback to default backend (CPU)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    return net
