### Lambda Notes


#### SSD

```
cd /DeepLearningExamples/PyTorch/Detection/SSD
docker build . -t nvidia_ssd

nvidia-docker run --rm -it --ulimit memlock=-1 --ulimit stack=67108864 -v /root/data/object_detection:/coco -v /root/DeepLearningExamples/PyTorch/Detection/SSD/results:/PyTorch_results --ipc=host nvidia_ssd

python -m torch.distributed.launch --nproc_per_node=8 \
       main.py --batch-size 128 \
               --mode benchmark-training \
               --benchmark-warmup 100 \
               --benchmark-iterations 200 \
               --amp \
               --data /coco |& tee /PyTorch_results/pytorch_Fluidstack_8xA100/ssd_amp.txt

python -m torch.distributed.launch --nproc_per_node=8 \
       main.py --batch-size 128 \
               --mode benchmark-training \
               --benchmark-warmup 100 \
               --benchmark-iterations 200 \
               --data /coco |& tee /PyTorch_results/pytorch_Fluidstack_8xA100/ssd_fp32.txt
```

#### ResNet50

```
cd /DeepLearningExamples/PyTorch/Classification/ConvNets
docker build . -t nvidia_rn50

nvidia-docker run --rm -it -v /root/data/imagenet:/data/imagenet -v /root//DeepLearningExamples/PyTorch/Classification/ConvNets/results:/PyTorch_results --ipc=host nvidia_rn50

python ./multiproc.py --nproc_per_node 8 \
       ./main.py --arch resnet50 \
       -b 256 \
       --training-only -p 1 \
       --raport-file benchmark.json \
       --epochs 1 \
       --amp --static-loss-scale 256 \
       --prof 100 /data/imagenet |& tee /PyTorch_results/pytorch_Fluidstack_8xA100/resnet50_amp.txt

python ./multiproc.py --nproc_per_node 8 \
       ./main.py --arch resnet50 \
       -b 256 \
       --training-only -p 1 \
       --raport-file benchmark.json \
       --epochs 1 \
       --prof 100 /data/imagenet |& tee /PyTorch_results/pytorch_Fluidstack_8xA100/resnet50_fp32.txt
```


#### BERT

```
cd /DeepLearningExamples/PyTorch/LanguageModeling/BERT
docker build --network=host . --rm --pull --no-cache -t bert


nvidia-docker run -it --rm \
  --gpus device=all \
  --net=host \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e LD_LIBRARY_PATH='/workspace/install/lib/' \
  -v /root/data/bert_large:/data/bert_large \
  -v /root/data/squad/v1.1:/data/squad/v1.1 \
  -v $PWD/results:/results \
  bert /bin/bash

bash scripts/run_squad.sh /data/bert_large/bert_large_uncased.pt 2.0 4 3e-5 fp16 8 42 /data/squad/v1.1 /data/bert_large/bert-large-uncased-vocab.txt /results/SQuAD train /data/bert_large/bert_config.json -1



```