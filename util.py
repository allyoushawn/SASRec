import sys
import copy
import random
import numpy as np
from collections import defaultdict, Counter


def get_item_prior(fname):
    f = open('data/%s.txt' % fname, 'r')
    item_occurence_list = []
    item_num = 0
    for line in f:
        _, i = line.rstrip().split(' ')
        i = int(i)
        item_num = max(i, item_num)
        item_occurence_list.append(i)
    counter = Counter(item_occurence_list)
    values = counter.values()
    sum_value = np.sum([x for x in values])
    item_prior = [counter[i] / sum_value for i in range(1, item_num + 1)]
    return item_prior

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

def retrieve_randomly(rated, itemnum):
    retrieved = []
    new_rated = set(rated)
    for _ in range(100):
        t = np.random.randint(1, itemnum + 1)
        while t in new_rated: t = np.random.randint(1, itemnum + 1)
        retrieved.append(t)
        new_rated.add(t)
    return retrieved

def retrieve_with_item_prior(rated, itemnum, item_prior):
    retrieved = []
    new_rated = set(rated)
    while len(retrieved) < 100:
        sampled_ids = np.random.choice(range(1, itemnum+1), 100, replace=False, p=item_prior)
        sampled_ids = [x for x in sampled_ids if x not in new_rated ]
        retrieved.extend(sampled_ids[:])
        new_rated.update(sampled_ids[:])
    return retrieved[:100]

def evaluate(model, dataset, args, item_prior=None):
    sess = model.sess
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(list(range(1, usernum + 1)), 10000)
    else:
        users = list(range(1, usernum + 1))
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        if item_prior is None:
            item_idx.extend(retrieve_randomly(rated, itemnum))
        else:
            item_idx.extend(retrieve_with_item_prior(rated, itemnum, item_prior))

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.'),
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, item_prior=None):
    sess = model.sess
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(list(range(1, usernum + 1)), 10000)
    else:
        users = list(range(1, usernum + 1))
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        if item_prior is None:
            item_idx.extend(retrieve_randomly(rated, itemnum))
        else:
            item_idx.extend(retrieve_with_item_prior(rated, itemnum, item_prior))

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.'),
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
