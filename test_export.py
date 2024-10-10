import pytest # 引入pytest包 
import os
import torch
from torch import optim, nn, utils
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import numpy as np
import onnxruntime

from ciallo_vit.model import LitAutoEncoder

def test_export_to_onnx():
    # init the autoencoder
    autoencoder = LitAutoEncoder()

    # setup data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((256, 256)),])
    dataset = MNIST(os.getcwd(), download=True, transform=transform)
    train_loader = utils.data.DataLoader(dataset)

    # train the model
    trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    # load checkpoint
    checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint)

    autoencoder.eval()
    # Specify the ONNX file path and export model
    filepath =  './lightning_logs/version_0/model.onnx'
    autoencoder.to_onnx(filepath, export_params=True, input_names=['input'], output_names=['output'], opset_version=18)

    # Load the ONNX model using ONNX Runtime
    ort_session = onnxruntime.InferenceSession(filepath)

    # Prepare input data
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: autoencoder.example_input_array.numpy()}

    # Get output from ONNX model
    ort_outs = ort_session.run(None, ort_inputs)

    # Check the output is the same as the original model
    with torch.no_grad():
        model_result_in_torch = autoencoder(autoencoder.example_input_array.to(autoencoder.device)).cpu().numpy()

    # Validate the correctness of the ONNX model
    is_valid = np.allclose(ort_outs[0], model_result_in_torch, atol=1e-6)

    assert is_valid, "The ONNX model is not valid"

if __name__ == '__main__':
    test_export_to_onnx()