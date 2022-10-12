File descriptions:
- dataloader.py: Init pytorch training dataset
- gat.py, gcn.py, gin.py, graphsage.py: GNN models
- main.py: Training offline blackbox models w/o online
- mlp.py: MLP models using for mlp students
- online_kd.py: Training online mode including a teacher (blackbox GNN) & students (Learners ~ Explainers)
- teacher.py: how to formulate a teacher in training. A naive way is just using the blackbox model
- utils.py: utility functions using throughout source codes

Training guidelines:

- MUTAG:
```
python online_kd.py --dataset Mutagenicity --device 0 --graph_pooling_type max --model_name gcn --linear_pooling_type last --epochs 200 \
                    --kd_strategy naive --norm_type bn --store_data ./ckpt/gcn/pgmutag_rand08.dat --datapath ./datasets/dgl_mutagenicity.pkl \
                    --split_name rand --split_ratio 0.8 --lr 0.01 --beta 5. --mk_term 0.001 --sl_term 0.0001 --model_path ./ckpt/gcn/pgmutag_rand5.pt
```

- BA2Motifs:
```
python online_kd.py --dataset BA --device 0 --graph_pooling_type max --model_name gcn --linear_pooling_type last --epochs 200 \
                    --kd_strategy naive --norm_type bn  --store_data ./ckpt/gcn/ba_f5.dat --split_name fold5 --lr 0.01 \
                    --model_path ./ckpt/gcn/ba.pt --kl_term 4 --beta 5 --use_norm_adj
```

Notebooks: the folder includes all notebooks using for execute models on test sets and evaluate explanations on evaluation sets