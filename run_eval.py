import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from util import evaluate, evaluate_valid, data_partition, get_item_prior


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

item_prior = get_item_prior(args.dataset)
dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = int(len(user_train) / args.batch_size)
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.disable_eager_execution()
sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = Model(sess, usernum, itemnum, args)
model.restore_vars('./test_save_model/my_model-1002')

T = 0.0
t0 = time.time()


try:
    t1 = time.time() - t0
    T += t1
    print ('Evaluating',)
    t_test = evaluate(model, dataset, args, item_prior=item_prior)
    t_valid = evaluate_valid(model, dataset, args, item_prior=item_prior)
    print ('')
    print ('time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
    T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
    f.write(str(t_valid) + ' ' + str(t_test) + '\n')
    f.flush()
    t0 = time.time()
except:
    sampler.close()
    f.close()
    exit(1)

f.close()
sampler.close()
print("Done")
