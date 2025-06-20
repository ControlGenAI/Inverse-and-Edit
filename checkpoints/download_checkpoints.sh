wget -P ./checkpoints https://storage.yandexcloud.net/yandex-research/invertible-cd/sd15_cfg_distill.pt.tar.gz
wget -P ./checkpoints https://storage.yandexcloud.net/yandex-research/invertible-cd/iCD-SD15_4steps_1.tar.gz
wget -P ./checkpoints https://huggingface.co/ilushado/Cycle-consistency/resolve/main/finetuned_forward_model.safetensors
tar -xvzf ./checkpoints/iCD-SD15_4steps_1.tar.gz  -C ./checkpoints && rm -rf ./checkpoints/iCD-SD15_4steps_1.tar.gz
tar -xvzf ./checkpoints/sd15_cfg_distill.pt.tar.gz -C ./checkpoints && rm -rf ./checkpoints/sd15_cfg_distill.pt.tar.gz
