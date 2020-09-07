### Lambda Notes


#### SSD

```
nvidia-docker run --rm -it --ulimit memlock=-1 --ulimit stack=67108864 -v $COCO_DIR:/coco -v /root/DeepLearningExamples/PyTorch/Detection/SSD/results:/PyTorch_results --ipc=host nvidia_ssd


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