from pandas import NA
import numpy as np
import random
import time
from dateutil.parser import parse as dparse
from utils.mutation import *
import evolocity as evo
from Bio.Seq import Seq
from Bio import Seq, SeqIO


np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Coronavirus sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='cov',
                        help='Model namespace')
    parser.add_argument('--dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Training minibatch size')
    parser.add_argument('--n-epochs', type=int, default=11,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint')
    parser.add_argument('--evolocity', action='store_true',
                        help='Analyze evolocity')
    parser.add_argument('--velocity-score', type=str, default='lm',
                        help='Analyze evolocity') 
    parser.add_argument('--downsample', type=float, default=100.,
                        help='Percentage to uniformly downsample.')
    parser.add_argument('--wdownsample', type=float, default=100.,
                        help='Percentage to weightedly downsampling.')
    args = parser.parse_args()
    return args

def parse_gisaid(entry):
    fields = entry.split('|')

    type_id = fields[1].split('/')[1]

    if type_id in { 'bat', 'canine', 'cat', 'env', 'mink',
                    'pangolin', 'tiger' }:
        host = type_id
        country = 'NA'
        continent = 'NA'
    else:
        host = 'human'
        from utils.locations import country2continent
        country = type_id
        if type_id in country2continent:
            continent = country2continent[country]
        else:
            continent = 'NA'

    from utils.mammals import species2group

    date = fields[2]
    date = date.replace('00', '01')
    timestamp = time.mktime(dparse(date).timetuple())

    meta = {
        'gene_id': fields[1],
        'date': date,
        'timestamp': timestamp,
        'host': host,
        'group': species2group[host].lower(),
        'country': country,
        'continent': continent,
        'dataset': 'gisaid',
    }
    return meta

def process(fnames):
    seqs = {}
    for fname in fnames:
        if fname.endswith('fasta'):

            for record in SeqIO.parse(fname, 'fasta'):
                if len(record.seq) !=201:
                    continue
                if str(record.seq).count('X') > 0:
                    continue
                rbd_seq=record.seq
                if rbd_seq not in seqs:
                    seqs[rbd_seq] = []
                meta = parse_gisaid(record.description)
                meta['accession'] = record.description
                seqs[rbd_seq].append(meta)
        elif fname.endswith("csv"):
            record_seqs=pd.read_csv(fname,index_col=0)["seq"].to_list()
            for record in record_seqs:
                record=Seq(record)
                if record not in seqs:
                    seqs[record]=[]
                meta = {
                    'gene_id': 'attack',
                    'date': '2022-04-08',
                    'timestamp': time.mktime(dparse('2022-04-08').timetuple()),
                    'host': 'attack',
                    'group': 'attack',
                    'country': 'attack',
                    'continent': 'attack',
                    'dataset': 'attack',
                    'accession': 'attack'
                }
                seqs[record].append(meta)
        else:
            raise TypeError("The file should be fatsa or csv")
    return seqs

def split_seqs(seqs, split_method='random'):
    train_seqs, test_seqs = {}, {}

    tprint('Splitting seqs...')
    for idx, seq in enumerate(seqs):
        if idx % 10 < 2:
            test_seqs[seq] = seqs[seq]
        else:
            train_seqs[seq] = seqs[seq]
    tprint('{} train seqs, {} test seqs.'
           .format(len(train_seqs), len(test_seqs)))

    return train_seqs, test_seqs

def setup(args):
    fnames = [
        "../data/RBD0308.fasta"
    ]

    import pickle
    cache_fname = '../result/target/ev_cache/cov_seqs.pkl'
    try:
        with open(cache_fname, 'rb') as f:
            seqs = pickle.load(f)
    except:
        seqs = process(fnames)
        with open(cache_fname, 'wb') as of:
            pickle.dump(seqs, of)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2
    model = get_model(args, seq_len, vocab_size,
                      inference_batch_size=1200)
    return model, seqs

def interpret_clusters(adata):
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        tprint('Cluster {}'.format(cluster))
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        for var in [ 'host', 'continent', 'gene_id' ]:
            tprint('\t{}:'.format(var))
            counts = Counter(adata_cluster.obs[var])
            for val, count in counts.most_common():
                tprint('\t\t{}: {}'.format(val, count))
        tprint('')

def plot_umap(adata, namespace='cov'):
    categories = [
        'timestamp',
        'louvain',
        'seqlen',
        'continent',
        'n_seq',
        'host',
    ]
    for category in categories:
        sc.pl.umap(adata, color=category,
                   edges=True, edges_color='#aaaaaa',
                   save='_{}_{}.png'.format(namespace, category))

def seqs_to_anndata(seqs):
    X, obs = [], {}
    obs['n_seq'] = []
    obs['seq'] = []
    obs['seqlen'] = []
    for seq in seqs:
        meta = seqs[seq][0]
        X.append(meta['embedding'])
        earliest_idx = np.argmin([
            meta['timestamp'] for meta in seqs[seq]
        ])
        for key in meta:
            if key == 'embedding':
                continue
            if key not in obs:
                obs[key] = []
            obs[key].append([
                meta[key] for meta in seqs[seq]
            ][earliest_idx])
        obs['n_seq'].append(len(seqs[seq]))
        obs['seq'].append(str(seq))
        obs['seqlen'].append(len(seq))
    X = np.array(X)
    adata = AnnData(X)
    for key in obs:
        adata.obs[key] = obs[key]

    return adata

def spike_evolocity(args, model, seqs, vocabulary, namespace='cov'):
    if args.model_name != 'esm1b':
        namespace += f'_{args.model_name}'
    if args.velocity_score != 'lm':
        namespace += f'_{args.velocity_score}'
    if args.downsample < 100:
        namespace += f'_downsample{args.seed}-{args.downsample}'
    elif args.wdownsample < 100:
        namespace += f'_wdownsample{args.seed}-{args.wdownsample}'

    ###############################
    ## Visualize Spike landscape ##
    ###############################

    adata_cache = '../result/target/ev_cache/cov_adata.h5ad'
    try:
        import anndata
        adata = anndata.read_h5ad(adata_cache)
    except:
        # seqs = embed_seqs(args, model, seqs, vocabulary,
        #                           use_cache=True)
        seqs = populate_embedding(args, model, seqs, vocabulary,
                                  use_cache=True)
        adata = seqs_to_anndata(seqs)
        adata = adata[
            adata.obs['timestamp'] >
            time.mktime(dparse('2019-11-30').timetuple())
        ]

        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
        sc.tl.louvain(adata, resolution=1.)
        sc.tl.umap(adata, min_dist=0.3)

        adata.write(adata_cache)

    adata.obs['seq'] = [ seq.rstrip('*') for seq in adata.obs['seq'] ]

    if args.downsample < 100:
        np.random.seed(args.seed)
        n_sample = round(len(adata) * (args.downsample / 100.))
        rand_idx = np.random.choice(len(adata), size=n_sample, replace=False)
        adata = adata[rand_idx]
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
        sc.tl.louvain(adata, resolution=1.)
        sc.tl.umap(adata, min_dist=1.)

    elif args.wdownsample < 100:
        np.random.seed(args.seed)
        n_sample = round(len(adata) * (args.wdownsample / 100.))
        # Upweight sequences more recent in time.
        weights = np.array(ss.rankdata(adata.obs['timestamp']))
        weights /= sum(weights)
        rand_idx = np.random.choice(
            len(adata), size=n_sample, replace=False, p=weights
        )
        adata = adata[rand_idx]
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
        sc.tl.louvain(adata, resolution=1.)
        sc.tl.umap(adata, min_dist=1.)

    if args.downsample < 100 or args.wdownsample < 100:
        with open(f'../result/target/ev_cache/{namespace}_rand_idx.txt', 'w') as of:
            of.write('\n'.join([ str(x) for x in rand_idx ]) + '\n')

    tprint('Analyzing {} sequences...'.format(adata.X.shape[0]))
    evo.set_figure_params(dpi_save=500)
    #plot_umap(adata, namespace=namespace)

    #####################################
    ## Compute evolocity and visualize ##
    #####################################

    cache_prefix = f'../result/target/ev_cache/{namespace}_knn30'
    try:
        from scipy.sparse import load_npz
        adata.uns["velocity_graph"] = load_npz(
            '{}_vgraph.npz'.format(cache_prefix)
        )
        adata.uns["velocity_graph_neg"] = load_npz(
            '{}_vgraph_neg.npz'.format(cache_prefix)
        )
        adata.obs["velocity_self_transition"] = np.load(
            '{}_vself_transition.npy'.format(cache_prefix)
        )
        adata.layers["velocity"] = np.zeros(adata.X.shape)
    except:
        evo.tl.velocity_graph(adata, model_name=args.model_name,
                              score=args.velocity_score)
        from scipy.sparse import save_npz
        save_npz('{}_vgraph.npz'.format(cache_prefix),
                 adata.uns["velocity_graph"],)
        save_npz('{}_vgraph_neg.npz'.format(cache_prefix),
                 adata.uns["velocity_graph_neg"],)
        np.save('{}_vself_transition.npy'.format(cache_prefix),
                adata.obs["velocity_self_transition"],)


if __name__ == '__main__':
    args = parse_args()

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
    ]
    vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) }

    model, seqs = setup(args)

    if 'esm' in args.model_name:
        vocabulary = { tok: model.alphabet_.tok_to_idx[tok]
                       for tok in model.alphabet_.tok_to_idx
                       if '<' not in tok and tok != '.' and tok != '-' }
        args.checkpoint = args.model_name
    elif args.model_name == 'tape':
        vocabulary = { tok: model.alphabet_[tok]
                       for tok in model.alphabet_ if '<' not in tok }
        args.checkpoint = args.model_name
    elif args.checkpoint is not None:
        model.model_.load_weights(args.checkpoint)
        tprint('Model summary:')
        tprint(model.model_.summary())

    if args.evolocity:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        spike_evolocity(args, model, seqs, vocabulary,
                        namespace=args.namespace)
