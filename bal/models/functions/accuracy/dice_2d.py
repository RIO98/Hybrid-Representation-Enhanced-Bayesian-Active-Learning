import numpy as np


def dice_score_2d(y_true, y_pred, n_classes=5):
    dice_scores = []

    for class_idx in range(1, n_classes):
        true_class = (y_true == class_idx).astype(np.float32)
        pred_class = (y_pred == class_idx).astype(np.float32)

        if np.sum(true_class) == 0 and np.sum(pred_class) == 0:
            dice_scores.append(1.0)
        else:
            intersection = np.sum(true_class * pred_class)
            union = np.sum(true_class) + np.sum(pred_class)
            dice = (2 * intersection) / (union + 1e-7)
            dice_scores.append(dice)

    return np.mean(dice_scores)


if __name__ == '__main__':
    n = 10
    h = 256
    w = 256

    y_true = np.random.randint(0, 5, (n, h, w))
    y_pred = np.random.randint(0, 5, (n, h, w))

    dice_scores = dice_score_2d(y_true, y_pred)
    print("Dice scores per class (1 to 4):", dice_scores)
