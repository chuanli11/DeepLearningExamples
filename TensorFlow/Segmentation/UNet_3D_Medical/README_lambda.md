Note
====


__Install__


```
git clone https://github.com/chuanli11/DeepLearningExamples.git
cd DeepLearningExamples/TensorFlow/Segmentation/UNet_3D_Medical

docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm --ipc=host -v ${PWD}:/workspace -v ${PWD}/data:/data -v ${PWD}/results:/results nvcr.io/nvidia/tensorflow:20.12-tf1-py3 /bin/bash
```

__Run__


```
bash examples/unet3d_debug_train_benchmark_TF-AMP.sh 1 /data/syn /results 2


bash examples/unet3d_debug_train_benchmark.sh 1 /data/syn /results 2
```
