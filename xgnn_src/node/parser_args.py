def add_common_args(parser):
    parser.add_argument("--dataset", type=str, required=False,
        help="cora|citeseer|pubmed|yelp|amazon")
    parser.add_argument("--graph-type", type=str, required=False,
        help="net_rur (for yelp)|net_upu (for amazon)|homo")
    parser.add_argument("--teacher-name", type=str, default="appnp",
            help="appnp|graphsage|gat")
    parser.add_argument("--teacher-pretrain", type=str,
            help="Teacher pretrained model")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--student-type", type=str, default="lpa",
                        help="lpa|sgat|mlp (when offline: use for specific student, when online: use for graph explainer)")
    parser.add_argument("--n-hidden", type=int, default=32,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--eval-type", type=str, default="acc")
    parser.add_argument("--skip-features", action="store_true",
            help="Skip features during training")
    parser.add_argument("--skip-structure", action="store_true",
            help="Skip structure during training")
    parser.add_argument("--all-layer-dp", action="store_true",
            help="Use dropout in all layers (Use for BA datasets)")
    parser.add_argument("--skip-norm", action="store_true",
            help="Skip batch norm in graph layers")
    parser.add_argument("--add-reverse", action="store_true",
            help="Add reversal graph to train")
    parser.set_defaults(self_loop=False)

def add_student_lpa_args(parser):
    # param for gcn-lpa
    parser.add_argument("--n-classes", type=int, default=-1)
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--std-n-hidden", type=int, default=32,
            help="number of hidden for student graph")
    parser.add_argument("--std-hiddens", type=int, nargs="+", default=[64],
            help="hidden sizes for student mlp")
    
    parser.add_argument("--surrogate", action='store_true',
            help="surrogate (default=False)")
    parser.add_argument("--n-lpa", type=int, default=10)
    parser.add_argument("--lpa-factor", type=float, default=1.,
            help="multiply to lpa loss")
    parser.add_argument("--slb", type=float, default=0.1,
            help="soft label factor in propagate")
    parser.add_argument("--lpm", type=str, default='random',
            help="initialization for lp adj random|norm")
    
    # param for surrogate model in soft entropy
    parser.add_argument("--temp", type=float, default=2,
            help="Soft cross entropy temperature")
    parser.add_argument("--sl-factor", type=float, default=1.,
            help="Soft loss factor")

    parser.add_argument("--student-graph", type=str,
            help="Store student graph")
    parser.add_argument("--student-model", type=str,
            help="Store student model")

    parser.add_argument("--batchsize", type=int, default=128,
            help="for mlp")
    parser.add_argument("--selected-feats", type=int, nargs="+", help="selected features to train")
    parser.add_argument("--std-emb", action="store_true", help="store student embeddings")


def add_appnp_args(parser):
    # param for appnp
    parser.add_argument("--hidden-sizes", type=int, nargs='+', default=[64],
            help="hidden unit sizes for appnp")
    parser.add_argument("--n-iter", type=int, default=20,
            help="num of iteration for appnp")

def add_gat_args(parser):
    # param fot gat
    parser.add_argument("--n-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--n-heads", type=int, default=3,
            help="number of heads")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")