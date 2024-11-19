## 準備

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 作業手順

GitHubを使ってコードの管理をしています。
作業する前と後には、以下のコマンドをターミナルで実行してください。
なんかエラーが出て進まない場合は、最悪無視でよいです。どこかのタイミングで直します。

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
# 1. Informerモデルの準備

cd Informer2020
# コマンドを実行して、Informerの学習をする
# Informerの学習パラメータはここで設定する
# ETTh1
python -u main_informer.py --model informer --data ETTh1 --attn prob --freq h --features S
# NaturalGas
python -u main_informer.py --model informer --data NaturalGas --root_path ./data/NaturalGas/ --attn prob --freq h --features S

cd ..

mkdir checkpoints
mv Informer2020/checkpoints/{学習済みモデルのファイル} checkpoints/{学習済みモデルのファイル}

# 2. main.ipynbの実行

一番上のセルの中身を適切に設定してから実行する
```
