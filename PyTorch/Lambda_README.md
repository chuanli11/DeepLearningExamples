### Lambda Notes


#### SSD

```
cd DeepLearningExamples/PyTorch/Detection/SSD

docker build . -t nvidia_ssd

HOST_DATA_DIR=/media/ubuntu/Data/coco
CONTAINER_DATA_DIR=/coco
HOST_RESULT_DIR=/home/ubuntu/DeepLearningExamples/PyTorch/Detection/SSD/results
CONTAINER_RESULT_DIR=/PyTorch_results

docker run --gpus "0" --rm -it --ulimit memlock=-1 --ulimit stack=67108864 -v $HOST_DATA_DIR:$CONTAINER_DATA_DIR -v $HOST_RESULT_DIR:$CONTAINER_RESULT_DIR --ipc=host nvidia_ssd

CONFIG=pytorch_quadrortx8000
mkdir -p /PyTorch_results/${CONFIG}

python -m torch.distributed.launch --nproc_per_node=1 \
       main.py --batch-size 32 \
               --mode benchmark-training \
               --benchmark-warmup 100 \
               --benchmark-iterations 200 \
               --amp \
               --data /coco |& tee /PyTorch_results/${CONFIG}/ssd_amp.txt

python -m torch.distributed.launch --nproc_per_node=1 \
       main.py --batch-size 32 \
               --mode benchmark-training \
               --benchmark-warmup 100 \
               --benchmark-iterations 200 \
               --data /coco |& tee /PyTorch_results/${CONFIG}/ssd_fp32.txt
```

#### ResNet50

```
cd DeepLearningExamples/PyTorch/Classification/ConvNets

docker build . -t nvidia_rn50

HOST_DATA_DIR=/media/ubuntu/Data1/data/imagenet
CONTAINER_DATA_DIR=/data/imagenet
HOST_RESULT_DIR=/home/ubuntu/DeepLearningExamples/PyTorch/Classification/ConvNets/results
CONTAINER_RESULT_DIR=/PyTorch_results


docker run --gpus "0" --rm -it -v $HOST_DATA_DIR:$CONTAINER_DATA_DIR -v $HOST_RESULT_DIR:$CONTAINER_RESULT_DIR --ipc=host nvidia_rn50

CONFIG=pytorch_quadrortx8000
mkdir -p /PyTorch_results/${CONFIG}

python ./multiproc.py --nproc_per_node 1 \
       ./main.py --arch resnet50 \
       -b 96 \
       --training-only -p 1 \
       --raport-file benchmark.json \
       --epochs 1 \
       --amp --static-loss-scale 256 \
       --prof 100 /data/imagenet |& tee /PyTorch_results/${CONFIG}/resnet50_amp.txt


python ./multiproc.py --nproc_per_node 1 \
       ./main.py --arch resnet50 \
       -b 48 \
       --training-only -p 1 \
       --raport-file benchmark.json \
       --epochs 1 \
       --prof 100 /data/imagenet |& tee /PyTorch_results/${CONFIG}/resnet50_fp32.txt
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