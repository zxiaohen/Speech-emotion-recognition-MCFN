def evaluate_metrics(pred_label, true_label):
    pred_label = np.array(pred_label)
    true_label = np.array(true_label)
    wa = np.mean(pred_label.astype(int) == true_label.astype(int))
    pred_onehot = np.eye(4)[pred_label.astype(int)]
    true_onehot = np.eye(4)[true_label.astype(int)]
    ua = np.mean(np.sum((pred_onehot == true_onehot) * true_onehot, axis=0) / np.sum(true_onehot, axis=0))
    key_metric, report_metric = 0.9 * wa + 0.1 * ua, {'wa': wa, 'ua': ua}
    return key_metric, report_metric
