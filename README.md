## 配置环境
```bash
conda env create -f environment.yml


推荐永远在项目根（...\你的文件>）跑：

conda activate jlr_hackathon
# 1) 生成合成数据
python scripts/make_fake_data.py

# 2) 训练树模型（RF/XGBoost）
python scripts/train_models.py

# 3) CodeBERT 微调（需网络）
python scripts/finetune_defect_easy.py   # 或 finetune_defect.py

# 4）GNN正在路上
