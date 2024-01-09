<div align="center">
  
# BackBone_tvr
</div>

To facilitate video retrieval research, I built a basic framework, which is based on fine-grained interactive retrieval of [wti](https://github.com/foolwood/DRL).

## Requirement
```shell
conda create -n tvr_env python=3.9
conda activate tvr_env
pip install -r requirements.txt
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

## Download CLIP Model
```shell
cd backbone_tvr/tvr/models
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
# wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

## Data Preparing
See [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) for details.

## Running
### MSRVTT
```sh
job_name="Project Name"
DATA_PATH="Your MSRVTT data and videos path"

CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
--master_port=2502 \
--nproc_per_node=1 \
main_retrieval.py \
--do_train=1 \
--workers=8 \
--n_display=10 \
--epochs=5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 64 \
--batch_size_val 16 \
--anno_path ${DATA_PATH}/MSRVTT/msrvtt_data \
--video_path ${DATA_PATH}/MSRVTT/MSRVTT/videos/all \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--base_encoder ViT-B/32 \
--output_dir ckpts/${job_name}
```

## Acknowledgments
The code is based on [DRL](https://github.com/foolwood/DRL) and [HBI](https://github.com/jpthu17/HBI). We sincerely appreciate for their contributions.
