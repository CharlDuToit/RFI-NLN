import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve, precision_recall_curve
import time


def recall_precision_fpr(y_true, y_pred, threshold):
    """
    y_pred is assumed to be binary or exclusively 0s and 1s
    y_pred and y_true must already be flattened
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred >= threshold).ravel()
    recall_ = tp / (tp + fn)
    precision_ = tp / (tp + fp) if tp+fp > 0.0 else 0.0
    fpr = fp / (tn + fp)
    # f1_ = 2 * precision_ * recall_ / (precision_ + recall_)

    return recall_, precision_, fpr


# def append_vals(recall, prec, fpr, thr, threshold)

def adaptive_threshold(y_true, y_pred, tol=0.01):
    start = time.time()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    min_thr = y_pred.min()
    max_thr = y_pred.max()

    recall_vals, prec_vals, fpr_vals, thr_vals = [0], [0], [0], [0.0]
    recall_vals[0], prec_vals[0], fpr_vals[0] = recall_precision_fpr(y_true, y_pred, 0.0)

    if min_thr == 0.0:
        index = 0
    else:
        thr_vals.append(min_thr)
        r, p, f = recall_precision_fpr(y_true, y_pred, min_thr)
        recall_vals.append(r)
        prec_vals.append(p)
        fpr_vals.append(f)
        index = 1

    if max_thr != 1.0:
        thr_vals.append(max_thr)
        r, p, f = recall_precision_fpr(y_true, y_pred, max_thr)
        recall_vals.append(r)
        prec_vals.append(p)
        fpr_vals.append(f)
        last_index = -2

    if max_thr == 1.0:
        last_index = -2 # should logically be -1

    thr_vals.append(1.0)
    r, p, f = recall_precision_fpr(y_true, y_pred, 1.0)
    recall_vals.append(r)
    prec_vals.append(p)
    fpr_vals.append(f)


    # do the stuff
    # index_hi = len(thr_vals) + last_index
    #prev_r = recall_vals[index]
    while index <= len(thr_vals) + last_index:
        thr = (thr_vals[index] + thr_vals[index+ 1])/2
        r, p, f = recall_precision_fpr(y_true, y_pred, thr)
        # if r != prev_r:
        if r not in recall_vals and p not in prec_vals and f not in fpr_vals:
            thr_vals.insert(index+1, thr)
            recall_vals.insert(index+1, r)
            prec_vals.insert(index+1, p)
            fpr_vals.insert(index+1, f)
            if np.abs(recall_vals[index] - r < tol):
                index += 1
        else:
            index += 1
        # prev_r = r

    print('adaptive curve time: ', time.time()-start)
    return recall_vals, prec_vals, fpr_vals, thr_vals


def sigmoid_recall_prec(y_true, y_pred):
    start = time.time()

    min_thr = y_pred.min()  # assume > 0
    max_thr = y_pred.max()  # asssume < 1
    recall_vals = []
    prec_vals = []
    thr_vals = []
    dec = count_decimals(min_thr)
    while dec > 1:
        y = np.round(np.clip(y_pred, 10**-dec, 10**(-dec+1)), dec+1)
        p, r, t = precision_recall_curve(y_true, y)  # at most 10 thresholds
        # print(r[1:-2])
        recall_vals += list(r[1:-2])
        prec_vals += list(p[1:-2])
        thr_vals += list(t[0:-2])
        # print('n_thr = ', len(thr_vals), ', n_unique_thr = ', len(np.unique(thr_vals)))
        # print('last threshold = ', thr_vals[-1])
        # print('n_recall = ', len(recall_vals), ', n_unique_recall = ', len(np.unique(recall_vals)))
        dec -= 1
    print('----------------------------')
    # print(recall_vals)
    # exit()
    y = np.round(np.clip(y_pred, 0.1, 0.9), 1+1)
    p, r, t = precision_recall_curve(y_true, y)  # at most 10 thresholds
    recall_vals += list(r[1:-2])
    prec_vals += list(p[1:-2])
    thr_vals += list(t[0:-2])
    max_dec = count_decimals(1 - max_thr)
    # dec should be 1 already
    while dec <= max_dec:
        y = np.round(np.clip(y_pred, 1 - 10**-dec, 1 - 10**(-dec-1)), dec+1+1)
        p, r, t = precision_recall_curve(y_true, y)  # at most 10 thresholds
        # print(r[1:-2])
        recall_vals += list(r[1:-2])
        prec_vals += list(p[1:-2])
        thr_vals += list(t[0:-2])
        # print('n_thr = ', len(thr_vals), ', n_unique_thr = ', len(np.unique(thr_vals)))
        # print('last threshold = ', thr_vals[-1])
        # print('n_recall = ', len(recall_vals), ', n_unique_recall = ', len(np.unique(recall_vals)))
        dec += 1

    # ind = len(prec_vals)-1
    prec_vals.append(1.0)
    recall_vals.append(0.0)
    thr_vals.append(1.0)
    print('adaptive curve time: ', time.time()-start)  # 0.5 for n_Val = 10000
    return recall_vals, prec_vals, thr_vals


def range_rounding(arr):
    # start = time.time()

    # print('Counting decimals....')
    min_thr = arr.min()  # assume > 0
    max_thr = arr.max()  # asssume < 1
    # Define the ranges and corresponding decimal places for rounding
    min_dec = count_decimals(min_thr)
    max_dec = count_decimals(1 - max_thr)
    # print('min_dec = ', min_dec, ', max_dec = ', max_dec)

    n_dec = 1

    ranges = []
    # ranges = [(1e-4, 1e-3, 5), (1e-3, 1e-2, 4)]
    for dec in range(min_dec, 1, -1):
        if dec == min_dec:
            ranges += [(0.0, 10**(-dec+1), dec+n_dec-1)]
        else:
            ranges += [(10**-dec, 10**(-dec+1), dec+n_dec-1)]

    ranges += [(0.1, 0.9, 2)]
    for dec in range(1, max_dec):
        if dec == max_dec -1:
            ranges += [(1 - 10**-dec, 1.0, dec+n_dec)]
        else:
            ranges += [(1 - 10 ** -dec, 1 - 10 ** (-dec - 1), dec+n_dec)]

    # print(ranges)

    for r in ranges:
        mask = (arr >= r[0]) & (arr < r[1])
        arr[mask] = np.round(arr[mask], decimals=r[2])

    # print('rounding time = ', time.time()-start)
    return arr

def count_decimals(number):
    if number == 0.0: return 30
    count = 0
    while np.abs(number) < 1.0:
        number *= 10
        count += 1
    return count

def sigmoid(x):
    return 1 / (1 + np.exp(x))


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    n_vals = int(1e7)
    np.random.seed(42)
    true = np.random.random((n_vals,)) > 0.5
    # pred = np.random.random((1000,))
    pred = sigmoid(np.random.uniform(-15, 15, size=(n_vals,)))

    #pred[0] = 0.0
    #pred[1] = 1.0
    #pred = np.clip(pred, 0.001, 0.01)

    print(pred.min(), pred.max())

    pred = range_rounding(pred)

    start = time.time()
    # recall_vals, prec_vals, fpr_vals, thr_vals = adaptive_threshold(true, pred, tol=0.01) # 1.98s for n_val = 10000, tol=0.01, n_thr=413
    prec_vals, recall_vals, thr_vals = precision_recall_curve(true, pred)  # 0.56s 10e6
    # prec_vals, recall_vals, thr_vals = precision_recall_curve(true, np.round(pred, 3))
    # prec_vals, recall_vals, thr_vals = precision_recall_curve(true, np.round(pred, 4))
    # recall_vals, prec_vals, thr_vals = sigmoid_recall_prec(true, pred) # ---- 3.01s 10e6
    print('standard curve time: ', time.time()-start)


    print('n_recall=', len(recall_vals) , ', n_thre=',len(thr_vals))
    print(auc(recall_vals, prec_vals))
    # print(auc(fpr_vals, recall_vals))
                                                                                        # with     pred[0] = 0.0
                                                                                        #          pred[1] = 1.0
    print(recall_vals[0:3], recall_vals[-3:]) # first element always = 1              # first two element  = 1
    print(prec_vals[0:3], prec_vals[-3:]) # last element always = 1 (not true)        # last two element = 1
    print(thr_vals[0:3], thr_vals[-3:]) # last threshold never = pred.max()           # first element always = pred.min()

    fig, ax = plt.subplots(figsize=(20,15))
    ax.plot(recall_vals, prec_vals, linewidth=1)
    ax.scatter(recall_vals, prec_vals, s=3)
    ax.set_ylim(top=0.7)
    ax.set_ylim(bottom=0.3)
    # ax.set_xlim(right=1.0)
    ax.set_xlim(right=0.3)
    ax.set_xlim(left=-0.01)
    plt.show()