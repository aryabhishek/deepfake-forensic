# deepfake-forensic

Model: facebook/deit-small-patch16-224

Detection type: Face + FFT (Fast Fourier Transform) dual-stream DeiT

Settings:
resolution: 224
batch: 16
mixed precision: ON
gradient accumulation: 2
Effective batch = 32

#https://huggingface.co/facebook/deit-small-patch16-224/blob/main/tf_model.h5

#dataset name : FaceForensics23++