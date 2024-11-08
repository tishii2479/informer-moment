if __name__=="__main__":
    e_layers = [1,2,3]
    d_layers = [1,2]
    dropout = [0.05,.1,0.3]
    learning_rate = [0.00001,0.0001,0.001]
    embed = ["timeF","learned"]
    for i in e_layers:
        for j in d_layers:
            for l in dropout:
                for k in learning_rate:
                    for m in embed:
                        file = f"e_layers{i}_d_layers{j}_dropout{l}_learning_rate{k}_embed{m}"
                        command = f"python -u main_informer.py --model informer --data ETTh1 --attn prob --freq h --features S  --e_layers {i}  --d_layers {j} --dropout {l} --learning_rate {k} --embed {m}| tee {file}.log"
                        print(command)

