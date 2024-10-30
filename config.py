import argparse

# 以下のコマンドを実行したときのargsの中身をコピー
# python -u main_informer.py --model informer --data ETTh1 --attn prob --freq h --features S
ARGS = argparse.Namespace(
    model="informer",
    data="ETTh1",
    root_path="./data/ETT/",
    data_path="ETTh1.csv",
    features="S",
    target="OT",
    freq="h",
    checkpoints="./checkpoints/",
    seq_len=96,
    label_len=48,
    pred_len=24,
    enc_in=1,
    dec_in=1,
    c_out=1,
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    s_layers=[3, 2, 1],
    d_ff=2048,
    factor=5,
    padding=0,
    distil=True,
    dropout=0.05,
    attn="prob",
    embed="timeF",
    activation="gelu",
    output_attention=False,
    do_predict=False,
    mix=True,
    cols=None,
    num_workers=0,
    itr=2,
    train_epochs=6,
    batch_size=32,
    patience=3,
    learning_rate=0.0001,
    des="test",
    loss="mse",
    lradj="type1",
    use_amp=False,
    inverse=False,
    use_gpu=False,
    gpu=0,
    use_multi_gpu=False,
    devices="0,1,2,3",
    detail_freq="h",
)
