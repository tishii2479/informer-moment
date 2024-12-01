## 準備

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 作業手順

GitHubを使ってコードの管理をしています。
作業する前と後には、以下のコマンドをターミナルで実行してください。
なんかエラーが出て進まない場合は、最悪無視でよいです。

```
# 作業前
git pull origin main

# 作業後
git add .
git commit -m "作業内容を表すメッセージをここに入れる、わかればなんでもよい"
git push origin main
```

## 実験手順

```
# ETTh1
python main.py --model informer --data ETTh1 --attn prob --freq h --features S  --e_layers 1 --d_layers 1 --dropout 0.3 --learning_rate 0.0001 --embed timeF --use_y_pred_cache --proposed_lmda 0.5 --proposed_moe_lr=0.1 --proposed_moe_weight_decay=0.01 --proposed_moe_epochs=20

# Natural Gas
python main.py --model informer --data NaturalGas --root_path './Informer2020/data/NaturalGas/' --data_path combined_data.csv --features S --attn prob --freq h --e_layers 1 --d_layers 1 --dropout 0.3 --learning_rate 0.0001 --embed timeF --use_y_pred_cache --proposed_lmda 0.4 --proposed_moe_lr=0.01 --proposed_moe_weight_decay=0.01 --proposed_moe_epochs=20
```
