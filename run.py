# -*- coding: utf-8 -*-

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Uncomment this to run the code on cpu
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
import argparse
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from model_core.config import ModelParam
from data_factory import LoadData, BatchLoader
from model_core.state_model import ClusterStateRecognition
from model_core.models import EvoNet_TSC
import model_core.metrics as mt

datainfos = {
    'djia30':    [50,  5, 4, 3.0],
    'webtraffic':[12, 30, 1, 3.0],
    'netflow':   [15, 24, 2, 6.0],
    'clockerr':  [12,  4, 2, 6.0]
}

def main(dataname, gpu, n_splits=5, max_epochs=100, patience=5):
    # 1) Set up parameters
    params = ModelParam()
    params.data_name     = dataname
    params.his_len       = datainfos[dataname][0]
    params.segment_len   = datainfos[dataname][1]
    params.segment_dim   = datainfos[dataname][2]
    params.node_dim      = 2 * params.segment_len * params.segment_dim
    params.id_gpu        = str(gpu)
    params.pos_weight    = datainfos[dataname][3]
    params.learning_rate = 0.001

    os.environ["CUDA_VISIBLE_DEVICES"] = params.id_gpu         #Enabling GPU (Comment this to run the code on CPU)
    
    # 2) Load entire train+test
    loader = LoadData()
    loader.set_configuration(params)
    all_train_x, all_train_y, test_x, test_y = loader.fetch_data()
    print(f"train+val shape: {all_train_x.shape}, test shape: {test_x.shape}")

    # 3) K-Fold cross‐validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_val_f1  = []
    fold_val_auc = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_train_x), 1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        # split
        tr_x, tr_y = all_train_x[train_idx], all_train_y[train_idx]
        vl_x, vl_y = all_train_x[val_idx], all_train_y[val_idx]

        # state recognition *per fold* (no reuse of saved GMM)
        state_model = ClusterStateRecognition()
        state_model.set_configuration(params)
        state_model.build_model()
        state_model.fit(tr_x)
        tr_prob, tr_pat = state_model.predict(tr_x)
        vl_prob, vl_pat = state_model.predict(vl_x)

        # BatchLoaders
        tr_loader = BatchLoader(params.batch_size)
        tr_loader.load_data(tr_x, tr_y, tr_prob, tr_pat, shuffle=True)
        vl_loader = BatchLoader(params.batch_size)
        vl_loader.load_data(vl_x, vl_y, vl_prob, vl_pat, shuffle=False)

        # building & training EvoNet_TSC with early stopping
        tf.reset_default_graph()
        model = EvoNet_TSC()
        model.set_configuration(params)
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0),
            allow_soft_placement=True
        )

        with tf.Session(config=config) as sess:
            model.build_model(is_training=True)
            sess.run(tf.global_variables_initializer())

            best_f1 = 0.0
            wait    = 0
            history = {
            'train_loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []
            }
            for epoch in range(max_epochs):
                loss = model.fit(sess, tr_loader)
                history['train_loss'].append(loss)
                # validation metrics
                yv_pred, yv_prob = model.predict(sess, vl_loader)
                res = mt.predict_accuracy(vl_y[:, -1], yv_pred)
                auc = mt.predict_auc(vl_y[:, -1], yv_prob[:, 1])
                history['precision'].append(res['Precision'])
                history['accuracy'].append(res['Accuracy'])
                history['recall'].append(res['Recall'])
                history['f1'].append(res['F1'])
                history['auc'].append(auc)
                print(f"Epoch {epoch:2d}  loss {loss:.4f}  |  "
                      f"Val Acc {res['Accuracy']:.4f}, Prec {res['Precision']:.4f}, "
                      f"Rec {res['Recall']:.4f}, F1 {res['F1']:.4f}, AUC {auc:.4f}")

                # early stopping
                if res['F1'] > best_f1:
                    best_f1 = res['F1']
                    wait = 0
                    model.store(params.model_save_path, sess=sess)
                    print("  -> new best val F1, checkpoint saved.")
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"No improvement for {patience} epochs, stopping fold early.")
                        break

        with open(f'fold_{fold}_history.pkl', 'wb') as f:
            pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

        # reloading of best‐checkpoint and record fold metrics
        tf.reset_default_graph()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            model = EvoNet_TSC()
            model.set_configuration(params)
            model.build_model(is_training=False)
            model.restore(params.model_save_path, sess=sess)
            yv_pred, yv_prob = model.predict(sess, vl_loader)
            res = mt.predict_accuracy(vl_y[:, -1], yv_pred)
            auc = mt.predict_auc(vl_y[:, -1], yv_prob[:, 1])
            fold_val_f1.append(res['F1'])
            fold_val_auc.append(auc)
            print(f">>> Fold {fold} best-on-val F1: {res['F1']:.4f}, AUC: {auc:.4f}")

    # Summarize cross-val results
    print("\n=== Cross-Validation Summary ===")
    print(f"Val F1:  {np.mean(fold_val_f1):.4f} ± {np.std(fold_val_f1):.4f}")
    print(f"Val AUC: {np.mean(fold_val_auc):.4f} ± {np.std(fold_val_auc):.4f}")

    # inal train on *all* training data, evaluate on held-out test set
    print("\n=== Final training on full train set & test evaluation ===")
    # re-fit state model on all_train_x
    state_model = ClusterStateRecognition()
    state_model.set_configuration(params)
    state_model.build_model()
    state_model.fit(all_train_x)
    all_prob, all_pat = state_model.predict(all_train_x)
    test_prob, test_pat = state_model.predict(test_x)

    tr_loader = BatchLoader(params.batch_size)
    tr_loader.load_data(all_train_x, all_train_y, all_prob, all_pat, shuffle=True)
    te_loader = BatchLoader(params.batch_size)
    te_loader.load_data(test_x, test_y, test_prob, test_pat, shuffle=False)

    tf.reset_default_graph()
    model = EvoNet_TSC()
    model.set_configuration(params)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model.build_model(is_training=True)
        sess.run(tf.global_variables_initializer())

        best_f1 = 0.0
        wait = 0
        for epoch in range(max_epochs):
            loss = model.fit(sess, tr_loader)
            yte_pred, yte_prob = model.predict(sess, te_loader)
            res = mt.predict_accuracy(test_y[:, -1], yte_pred)
            auc = mt.predict_auc(test_y[:, -1], yte_prob[:, 1])
            print(f"Epoch {epoch:2d}  loss {loss:.4f}  |  "
                  f"Test Acc {res['Accuracy']:.4f}, Prec {res['Precision']:.4f}, "
                  f"Rec {res['Recall']:.4f}, F1 {res['F1']:.4f}, AUC {auc:.4f}")
            if res['F1'] > best_f1:
                best_f1 = res['F1']
                wait = 0
                model.store(params.model_save_path, sess=sess)
            else:
                wait += 1
                if wait >= patience:
                    print(f"No improvement for {patience} epochs, stopping final train.")
                    break

    # final restore & report
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = EvoNet_TSC()
        model.set_configuration(params)
        model.build_model(is_training=False)
        model.restore(params.model_save_path, sess=sess)
        yte_pred, yte_prob = model.predict(sess, te_loader)
        res = mt.predict_accuracy(test_y[:, -1], yte_pred)
        auc = mt.predict_auc(test_y[:, -1], yte_prob[:, 1])
        print(f"\nFINAL TEST → Accuracy {res['Accuracy']:.4f}, "
              f"Precision {res['Precision']:.4f}, Recall {res['Recall']:.4f}, "
              f"F1 {res['F1']:.4f}, AUC {auc:.4f}")

def getattention(dataname, gpu=0):
    params = ModelParam()
    params.data_name = dataname
    params.his_len = datainfos[params.data_name][0]
    params.segment_len = datainfos[params.data_name][1]
    params.segment_dim = datainfos[params.data_name][2]
    params.node_dim = 2 * params.segment_dim * params.segment_len
    params.id_gpu = '{}'.format(gpu)
    params.pos_weight = datainfos[params.data_name][3]
    params.learning_rate = 0.001

    os.environ["CUDA_VISIBLE_DEVICES"] = params.id_gpu

    dataloader = LoadData()
    dataloader.set_configuration(params)

    trainx, trainy, _, _ = dataloader.fetch_data()
    rawx = trainx
    rawy = trainy
    print(rawx.shape, rawy.shape)

    # state
    print("state recognizing...")
    state_model = ClusterStateRecognition()
    state_model.set_configuration(params)
    state_model.build_model()
    prob, patterns = state_model.predict(rawx)
    print(patterns.shape, prob.shape)

    # establish dataloader
    testloader = BatchLoader(params.batch_size)
    testloader.load_data(rawx, rawy, prob, patterns, shuffle=False)

    # model
    model = EvoNet_TSC()
    model.set_configuration(params)
    tf.reset_default_graph()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model.build_model(is_training=False)
        model.restore(params.model_save_path, sess=sess)

        y_pred, y_pred_prob = model.predict(sess, testloader)
        attentions = model.getAttention(sess, testloader)

        results = mt.predict_accuracy(rawy[:, -1], y_pred)
        auc = mt.predict_auc(rawy[:, -1], y_pred_prob[:, 1])
        logstr = 'Accuracy {:f}, Precision {:f}, Recall {:f}, F1 {:f}, AUC {:f}'.format(results['Accuracy'], results['Precision'], results['Recall'], results['F1'], auc)
        print(logstr)

        store_obj = {'x': rawx, 'y': rawy, 'prob': prob, 'pattern':patterns, 'attention':attentions}
        pickle.dump(store_obj, open('./Repo/output/result_{}.pkl'.format(dataname), 'wb'), pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset",choices=list(datainfos.keys()),
                        default='djia30', help="select dataset")
    parser.add_argument("-g","--gpu",choices=['0','1','2'],default='0',
                        help="target gpu id")
    args = parser.parse_args()

    main(args.dataset, args.gpu)
    getattention(args.dataset, args.gpu)
    
