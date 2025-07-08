# # Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number

import numpy as np
import torch
from torch.nn.functional import one_hot
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skmet
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    confusion_matrix,
)
import seaborn as sns


def auc_and_roc_curve(
    lab_real, lab_pred, class_names, class_to_compute="all", save_path=None
):
    """
    This function computes the ROC curves and AUC for each class.
    It better described on: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    Both lab_real and lab_pred can be a labels array or and a array of scores (one hot encoding) for each class.
    :param lab_real (np.array): the data real labels
    :param lab_pred (np.array): the predictions returned by the model
    :param class_names (list): the name of each label. For example: ['l1','l2']. If you pass a list with a different
    :param class_to_compute (string, optional): select the class you'd like to compute the ROC. If you set 'all', it
    will compute all curves. Note that you should inform a valid class, that is, a class that is inside in class_name.
    Default is 'all'.
    :return: a dictionaty with the AUC, fpr, tpr for each class
    """

    # Checkin the array dimension
    # lab_real, lab_pred = _check_dim(lab_real, lab_pred, mode="scores")
    lab_real = one_hot(lab_real.long(), num_classes=len(class_names))
    lab_real = lab_real.numpy()

    # Computing the ROC curve and AUC for each class
    fpr = dict()  # false positive rate
    tpr = dict()  # true positive rate
    roc_auc = dict()  # area under the curve
    for i, name in enumerate(class_names):
        # print(i, name)
        fpr[name], tpr[name], _ = skmet.roc_curve(lab_real[:, i], lab_pred[:, i])
        roc_auc[name] = skmet.auc(fpr[name], tpr[name])

    if class_to_compute == "all":
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[name] for name in class_names]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for name in class_names:
            mean_tpr += np.interp(all_fpr, fpr[name], tpr[name])

        # Finally average it and compute AUC
        mean_tpr /= float(len(class_names))

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = skmet.auc(fpr["macro"], tpr["macro"])

        # Computing the micro-average ROC curve and the AUC
        fpr["micro"], tpr["micro"], _ = skmet.roc_curve(
            lab_real.ravel(), lab_pred.ravel()
        )
        roc_auc["micro"] = skmet.auc(fpr["micro"], tpr["micro"])

        if save_path:
            # Ploting all ROC curves
            plt.figure()

            # Plotting the micro avg
            plt.plot(
                fpr["micro"],
                tpr["micro"],
                label="MicroAVG - AUC: {0:0.4f}" "".format(roc_auc["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=2,
            )

            # Plotting the micro avg
            plt.plot(
                fpr["macro"],
                tpr["macro"],
                label="MacroAVG - AUC: {0:0.4f}" "".format(roc_auc["macro"]),
                color="navy",
                linestyle=":",
                linewidth=2,
            )

            # Plottig the curves for each class
            for name in class_names:
                plt.plot(
                    fpr[name],
                    tpr[name],
                    linewidth=1,
                    label="{0} - AUC: {1:0.4f}" "".format(name, roc_auc[name]),
                )

    else:
        if save_path:
            plt.plot(
                fpr[class_to_compute],
                tpr[class_to_compute],
                linewidth=1,
                label="{0} - AUC: {1:0.4f}"
                "".format(class_to_compute, roc_auc[class_to_compute]),
            )

    if save_path:
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curves")
        plt.legend(loc="lower right")

        if isinstance(save_path, str):
            plt.savefig(save_path)
            plt.clf()
        elif save_path:
            plt.show()

    return roc_auc["micro"]


def plot_confusion_matrix(cm, savename, title="Confusion Matrix"):
    if len(cm) == 1:
        classes = ["Tb", "Tg"]
        cm = cm[0]
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 8), dpi=600)
        np.set_printoptions(precision=2)

        # 在混淆矩阵中每格的概率值
        ind_array = np.arange(len(classes))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            if c > 0.001:
                plt.text(
                    x_val,
                    y_val,
                    "%0.2f" % (c,),
                    color="red",
                    fontsize=15,
                    va="center",
                    ha="center",
                )

        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.binary)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(classes)))
        plt.xticks(xlocations, classes, rotation=90)
        plt.yticks(xlocations, classes)
        plt.ylabel("Actual label")
        plt.xlabel("Predict label")

        # offset the tick
        tick_marks = np.array(range(len(classes))) + 0.5
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position("none")
        plt.gca().yaxis.set_ticks_position("none")
        plt.grid(True, which="minor", linestyle="-")
        plt.gcf().subplots_adjust(bottom=0.15)

        # show confusion matrix
        plt.savefig(savename, format="png")
        plt.close()
    else:
        num_cms = len(cm)
        plt.figure(figsize=(5 * num_cms, 5), dpi=600)
        for index, c in enumerate(cm):
            if index == 0:
                title = "Confusion Matrix" + " FUSE"
            elif index == 1:
                title = "Confusion Matrix" + " COR"
            elif index == 2:
                title = "Confusion Matrix" + " SAG"
            else:
                title = "Confusion Matrix %d" % index
            if c.shape[0] == 5:
                classes = ["T0", "T1", "T2a", "T2b", "T3"]
            elif c.shape[0] == 4:
                classes = ["T0/1", "T2a", "T2b", "T3"]
            elif c.shape[0] == 2:
                classes = ["Tb", "Tg"]
            else:
                assert False, "make sure the output class num if 4 or 5 or 2"
            c = c.astype("float") / c.sum(axis=1)[:, np.newaxis]
            plt.subplot(1, num_cms, index + 1)
            np.set_printoptions(precision=2)
            # 在混淆矩阵中每格的概率值
            ind_array = np.arange(len(classes))
            x, y = np.meshgrid(ind_array, ind_array)
            for x_val, y_val in zip(x.flatten(), y.flatten()):
                ct = c[y_val][x_val]
                if ct > 0.001:
                    plt.text(
                        x_val,
                        y_val,
                        "%0.2f" % (ct,),
                        color="red",
                        fontsize=15,
                        va="center",
                        ha="center",
                    )

            plt.imshow(c, interpolation="nearest", cmap=plt.cm.binary)
            plt.title(title)
            plt.colorbar()
            xlocations = np.array(range(len(classes)))
            plt.xticks(xlocations, classes, rotation=90)
            plt.yticks(xlocations, classes)
            plt.ylabel("Actual label")
            plt.xlabel("Predict label")

            # offset the tick
            tick_marks = np.array(range(len(classes))) + 0.5
            plt.gca().set_xticks(tick_marks, minor=True)
            plt.gca().set_yticks(tick_marks, minor=True)
            plt.gca().xaxis.set_ticks_position("none")
            plt.gca().yaxis.set_ticks_position("none")
            plt.grid(True, which="minor", linestyle="-")
            plt.gcf().subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.savefig(savename, format="png")
        plt.close()


def calculate_confusion_matrix(pred, target):
    """Calculate confusion matrix according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        torch.Tensor: Confusion matrix
            The shape is (C, C), where C is the number of classes.
    """

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor), (
        f"pred and target should be torch.Tensor or np.ndarray, "
        f"but got {type(pred)} and {type(target)}."
    )

    # Modified from PyTorch-Ignite
    num_classes = pred.size(1)
    pred_label = torch.argmax(pred, dim=1).flatten()
    target_label = target.flatten()
    assert len(pred_label) == len(target_label)

    with torch.no_grad():
        indices = num_classes * target_label + pred_label
        matrix = torch.bincount(indices, minlength=num_classes**2)
        matrix = matrix.reshape(num_classes, num_classes)
    return matrix


def precision_recall_f1(pred, target, average_mode="macro", thrs=0.0):
    """Calculate precision, recall and f1 score according to the prediction and
    target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        tuple: tuple containing precision, recall, f1 score.

            The type of precision, recall, f1 score is one of the following:

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ["macro", "none"]
    if average_mode not in allowed_average_mode:
        raise ValueError(f"Unsupport type of averaging {average_mode}.")

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    assert isinstance(
        pred, torch.Tensor
    ), f"pred should be torch.Tensor or np.ndarray, but got {type(pred)}."
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).long()
    assert isinstance(target, torch.Tensor), (
        f"target should be torch.Tensor or np.ndarray, " f"but got {type(target)}."
    )

    if isinstance(thrs, Number):
        thrs = (thrs,)
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(f"thrs should be a number or tuple, but got {type(thrs)}.")

    num_classes = pred.size(1)
    pred_score, pred_label = torch.topk(pred, k=1)
    pred_score = pred_score.flatten()
    pred_label = pred_label.flatten()

    gt_positive = one_hot(target.flatten(), num_classes)

    precisions = []
    recalls = []
    f1_scores = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        pred_positive = one_hot(pred_label, num_classes)
        if thr is not None:
            pred_positive[pred_score <= thr] = 0
        class_correct = (pred_positive & gt_positive).sum(0)
        precision = class_correct / np.maximum(pred_positive.sum(0), 1.0) * 100
        recall = class_correct / np.maximum(gt_positive.sum(0), 1.0) * 100
        f1_score = (
            2
            * precision
            * recall
            / np.maximum(precision + recall, torch.finfo(torch.float32).eps)
        )
        if average_mode == "macro":
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        elif average_mode == "none":
            precision = precision.detach().cpu().numpy()
            recall = recall.detach().cpu().numpy()
            f1_score = f1_score.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupport type of averaging {average_mode}.")
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if return_single:
        return precisions[0], recalls[0], f1_scores[0]
    else:
        return precisions, recalls, f1_scores


def precision(pred, target, average_mode="macro", thrs=0.0):
    """Calculate precision according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Precision.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    precisions, _, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return precisions


def recall(pred, target, average_mode="macro", thrs=0.0):
    """Calculate recall according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Recall.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, recalls, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return recalls


def average_precision(pred, target):
    r"""Calculate the average precision for a single class.

    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:

    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    Note that no approximation is involved since the curve is piecewise
    constant.

    Args:
        pred (np.ndarray): The model prediction with shape (N, ).
        target (np.ndarray): The target of each prediction with shape (N, ).

    Returns:
        float: a single float as average precision value.
    """
    eps = np.finfo(np.float32).eps

    # sort examples
    sort_inds = np.argsort(-pred)
    sort_target = target[sort_inds]

    # count true positive examples
    pos_inds = sort_target == 1
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]

    # count not difficult examples
    pn_inds = sort_target != -1
    pn = np.cumsum(pn_inds)

    tp[np.logical_not(pos_inds)] = 0
    precision = tp / np.maximum(pn, eps)
    ap = np.sum(precision) / np.maximum(total_pos, eps)
    return ap


def mAP(pred, target):
    """Calculate the mean average precision with respect of classes.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.

    Returns:
        float: A single float as mAP value.
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError("pred and target should both be torch.Tensor or" "np.ndarray")

    assert pred.shape == target.shape, "pred and target should be in the same shape."
    num_classes = pred.shape[1]
    ap = np.zeros(num_classes)
    for k in range(num_classes):
        ap[k] = average_precision(pred[:, k], target[:, k])
    mean_ap = ap.mean() * 100.0
    return mean_ap


def accuracy_numpy(pred, target, topk=(1,), thrs=0.0):
    if isinstance(thrs, Number):
        thrs = (thrs,)
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(f"thrs should be a number or tuple, but got {type(thrs)}.")

    res = []
    maxk = max(topk)
    num = pred.shape[0]

    static_inds = np.indices((num, maxk))[0]
    pred_label = pred.argpartition(-maxk, axis=1)[:, -maxk:]
    pred_score = pred[static_inds, pred_label]

    sort_inds = np.argsort(pred_score, axis=1)[:, ::-1]
    pred_label = pred_label[static_inds, sort_inds]
    pred_score = pred_score[static_inds, sort_inds]

    for k in topk:
        correct_k = pred_label[:, :k] == target.reshape(-1, 1)
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct_k = correct_k & (pred_score[:, :k] > thr)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            res_thr.append((_correct_k.sum() * 100.0 / num))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy_torch(pred, target, topk=(1,), thrs=0.0):
    if isinstance(thrs, Number):
        thrs = (thrs,)
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(f"thrs should be a number or tuple, but got {type(thrs)}.")

    res = []
    maxk = max(topk)
    num = pred.size(0)
    pred = pred.float()
    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    for k in topk:
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct = correct & (pred_score.t() > thr)
            correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res_thr.append((correct_k.mul_(100.0 / num)))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy(pred, target, topk=1, thrs=0.0):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction
        topk (int | tuple[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]: Accuracy
            - torch.Tensor: If both ``topk`` and ``thrs`` is a single value.
            - list[torch.Tensor]: If one of ``topk`` or ``thrs`` is a tuple.
            - list[list[torch.Tensor]]: If both ``topk`` and ``thrs`` is a \
              tuple. And the first dim is ``topk``, the second dim is ``thrs``.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    assert isinstance(pred, (torch.Tensor, np.ndarray)), (
        f"The pred should be torch.Tensor or np.ndarray " f"instead of {type(pred)}."
    )
    assert isinstance(target, (torch.Tensor, np.ndarray)), (
        f"The target should be torch.Tensor or np.ndarray "
        f"instead of {type(target)}."
    )

    # torch version is faster in most situations.
    to_tensor = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    pred = to_tensor(pred)
    target = to_tensor(target)

    res = accuracy_torch(pred, target, topk, thrs)

    return res[0] if return_single else res


def get_metrics(
    gt_labels_list, results_list, index="", pred_logit=None, save_path=None, **kwargs
):
    assert len(results_list) == len(gt_labels_list), "GT and Pred shape do not match"
    num_samples = len(results_list)
    gt_array = np.asarray(gt_labels_list)
    pre_array = np.asarray(results_list)
    diff = abs(gt_array - pre_array)
    acc = np.where(diff == 0, 1, 0).sum() / num_samples
    acc_smooth = (
        np.where(diff == 1, 1, 0).sum() + np.where(diff == 0, 1, 0).sum()
    ) / num_samples

    cm = confusion_matrix(gt_labels_list, results_list)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    gt_array_fuse = np.where(gt_array == 0, 1, gt_array)
    pre_array_fuse = np.where(pre_array == 0, 1, pre_array)
    diff_fuse = abs(gt_array_fuse - pre_array_fuse)
    acc_fuse = np.where(diff_fuse == 0, 1, 0).sum() / num_samples
    acc_smooth_fuse = (
        np.where(diff_fuse == 1, 1, 0).sum() + np.where(diff_fuse == 0, 1, 0).sum()
    ) / num_samples
    if pred_logit is not None:
        class_names = ["Tb", "Tg"]
        # if pred_logit.shape[1] == 4:
        #     class_names = ["T0/1", "T2a", "T2b", "T3"]
        # elif pred_logit.shape[1] == 5:
        #     class_names = ["T0", "T1", "T2a", "T2b", "T3"]
        # elif pred_logit.shape[1] == 2:
        #     class_names = ["Tb", "Tg"]
        # else:
        #     assert False, "class number not right {}".format(pred_logit.shape[1])
        auc = auc_and_roc_curve(
            torch.Tensor(gt_array),
            torch.Tensor(pred_logit),
            class_names,
            save_path=save_path,
        )
    else:
        auc = 0.0
    results = {}
    results["acc%s" % index] = acc
    results["acc_smooth%s" % index] = acc_smooth
    results["acc_fuse%s" % index] = acc_fuse
    results["acc_smooth_fuse%s" % index] = acc_smooth_fuse
    results["precision%s" % index] = precision.mean()
    results["recall%s" % index] = recall.mean()
    results["f1%s" % index] = f1.mean()
    results["auc%s" % index] = auc
    return results


def save_confusion_matrix(
    y_true, y_pred, epoch, log_dir, phase, num_classes=2, task_type="binary"
):
    """
    绘制并保存混淆矩阵
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param epoch: 当前 epoch
    :param log_dir: 保存图像的文件夹
    :param num_classes: 类别数量，默认为二分类（2）
    :param task_type: 任务类型，'binary' 或 'multiclass'
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[f"Class {i}" for i in range(num_classes)],
        yticklabels=[f"Class {i}" for i in range(num_classes)],
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Epoch {epoch})")

    plt.savefig(f"{log_dir}/{phase}_cm_epoch{epoch}.png")
    plt.close()
    return cm


def calculate_and_save_metrics(
    y_true, logits, epoch, log_dir, phase, num_classes=2, threshold=0.5
):
    """
    计算指标（Acc, F1, AUC, AP, Recall等），并保存 ROC 和 PR 曲线图片
    :param y_true: 真实标签 (Numpy array)
    :param logits: 模型输出的 logits (Tensor)
    :param epoch: 当前 epoch
    :param log_dir: 保存文件的目录
    :param num_classes: 类别数量，默认为二分类（2）
    :param threshold: 二分类的预测阈值
    """
    y_true = np.array(y_true).reshape(-1)

    # === 确定任务类型 ===
    if logits.ndimension() == 1 or logits.size(1) == 1 or num_classes == 2:
        # 二分类
        task_type = "binary"
        probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()
        preds = (probs >= threshold).astype(int)
        auc_score = roc_auc_score(y_true, probs)
        ap = average_precision_score(y_true, probs)

        # === 绘制 ROC 和 PR 曲线 ===
        fpr, tpr, _ = roc_curve(y_true, probs)
        save_roc_curve(fpr, tpr, auc_score, epoch, log_dir, phase, task_type)

        precision, recall, _ = precision_recall_curve(y_true, probs)
        save_pr_curve(precision, recall, ap, epoch, log_dir, phase, task_type)
        recall = recall_score(y_true, preds, average="binary", zero_division=0)
        f1 = f1_score(y_true, preds, average="binary", zero_division=0)

    else:
        # 多分类
        task_type = "multiclass"
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        auc_score = roc_auc_score(y_true, probs, multi_class="ovo", average="macro")

        ap_scores = [
            average_precision_score((y_true == i).astype(int), probs[:, i])
            for i in range(probs.shape[1])
        ]
        ap = np.mean(ap_scores)

        # === 绘制 ROC 和 PR 曲线 ===
        for i in range(probs.shape[1]):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), probs[:, i])
            save_roc_curve(fpr, tpr, auc(fpr, tpr), epoch, log_dir, phase, task_type, i)

            precision, recall, _ = precision_recall_curve(
                (y_true == i).astype(int), probs[:, i]
            )
            save_pr_curve(
                precision, recall, ap_scores[i], epoch, log_dir, phase, task_type, i
            )
        recall = recall_score(y_true, preds, average="macro", zero_division=0)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    # === 计算常用指标 ===
    acc = accuracy_score(y_true, preds)

    # === 绘制并保存混淆矩阵 ===
    cm = save_confusion_matrix(
        y_true, preds, epoch, log_dir, phase, num_classes, task_type
    )
    metrics = {
        "acc": round(acc, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc_score, 4),
        "ap": round(ap, 4),
    }

    return metrics, cm


def save_roc_curve(
    fpr, tpr, auc_score, epoch, log_dir, phase, task_type, class_id=None
):
    """
    保存 ROC 曲线
    :param fpr: 假阳性率
    :param tpr: 真阳性率
    :param auc_score: AUC 分数
    :param epoch: 当前 epoch
    :param log_dir: 保存的文件夹
    :param phase: train or val
    :param task_type: 任务类型（'binary' 或 'multiclass'）
    :param class_id: 类别 id（仅多分类时需要）
    """
    plt.figure()
    if task_type == "binary":
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    else:
        plt.plot(fpr, tpr, label=f"Class {class_id} AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Epoch {epoch})")
    plt.legend(loc="lower right")
    plt.grid(True)
    if task_type == "binary":
        plt.savefig(f"{log_dir}/{phase}_roc_epoch{epoch}.png")
    else:
        plt.savefig(f"{log_dir}/{phase}_roc_class{class_id}_epoch{epoch}.png")
    plt.close()


def save_pr_curve(
    precision, recall, ap_score, epoch, log_dir, phase, task_type, class_id=None
):
    """
    保存 PR 曲线
    :param precision: 精度
    :param recall: 召回率
    :param ap_score: 平均精度
    :param epoch: 当前 epoch
    :param log_dir: 保存的文件夹
    :param phase: train or val
    :param task_type: 任务类型（'binary' 或 'multiclass'）
    :param class_id: 类别 id（仅多分类时需要）
    """
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap_score:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (Epoch {epoch})")
    plt.legend(loc="lower left")
    plt.grid(True)
    if task_type == "binary":
        plt.savefig(f"{log_dir}/{phase}_pr_epoch{epoch}.png")
    else:
        plt.savefig(f"{log_dir}/{phase}_pr_class{class_id}_epoch{epoch}.png")
    plt.close()
