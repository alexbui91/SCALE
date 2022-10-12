File descriptions:
- draw.py: utility functions for visualizing explanation graphs & ground-truth graphs
- eval.py: unitily functions for evaluate explanations
- online_kg2.py: main files for training blackbox w/ learners using online knowledge distillation
- parser_args.py: input arguments
- student_online.py: contains all student models for online mode
- student.py: contains all student models for offline mode (using for EGCN baseline). At first we developed an offline mode KD to varify whether KD can work on graph data. This motivates our proposed method (online KD) in this paper.
- teacher_online.py: contains all teacher models for online mode
- teacher.py: similar to student.py
- train.py: training offline mode (using for EGCN baseline)
- utils.py: utility functions using for node classification problems

Training guidelines:
- BA-Shapes
```
python online_kg2.py --dataset BAS --temp 2 --n-epochs 1000 --gpu 0 --teacher-name gcn2 --student-type gcn \
    --n-hidden 64 --n-layers 5 --sl-factor 0.1 --lr 0.01 --all-layer-dp \
    --skip-norm --teacher-pretrain ./ckpt/gcn/ba_shape_feat_dir.pt
```

If using reverse edges
```
python online_kg2.py --dataset BAS --temp 2 --n-epochs 1000 --gpu 0 --teacher-name gcn2 --student-type gcn \
    --n-hidden 64 --n-layers 5 --sl-factor 0.1 --lr 0.01 --all-layer-dp \
    --skip-norm --add-reverse --teacher-pretrain ./ckpt/gcn/ba_shape_feat_bidir.pt
```
- BA-Community
python online_kg2.py --dataset BAC --temp 2 --n-epochs 1000 --gpu 0 --teacher-name gcn2 --student-type gcn \
    --n-hidden 64 --n-layers 5 --sl-factor 0.1 --lr 0.01 --all-layer-dp \
    --skip-norm --teacher-pretrain ./ckpt/gcn/ba_community_feat_dir.pt

If using reverse edges
```
python online_kg2.py --dataset BAC --temp 2 --n-epochs 1000 --gpu 0 --teacher-name gcn2 --student-type gcn \
    --n-hidden 128 --n-layers 4 --sl-factor 0.1 --lr 0.01 --all-layer-dp --skip-norm --add-reverse \
    --teacher-pretrain ./ckpt/gcn/ba_community_feat_bidir.pt
```

- Tree-Cycle
```
python online_kg2.py --dataset TRC --temp 2 --n-epochs 1000 --gpu 0 --teacher-name gcn2 --student-type gcn \
    --n-hidden 64 --n-layers 5 --sl-factor 0.1 --lr 0.01 --all-layer-dp \
    --skip-norm --add-reverse --teacher-pretrain ./ckpt/gcn/tree_cycle_feat_bidir.pt
```

- Tree-Grid
```
 python online_kg2.py --dataset TRC --temp 2 --n-epochs 1000 --gpu 0 --teacher-name gcn2 --student-type gcn \
    --n-hidden 64 --n-layers 5 --sl-factor 0.1 --lr 0.01 --all-layer-dp \
    --skip-norm --add-reverse --teacher-pretrain ./ckpt/gcn/tree_cycle_feat_bidir.pt
```

Notebooks: the folder contains all notebooks using for executing models on test sets and evaluate explanations