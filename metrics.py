import numpy as np
import torch

def miou(pred_mask, true_mask, num_classes=2):
    class_iou = torch.zeros(num_classes)
    assert (pred_mask.shape == true_mask.shape)
    for class_idx in range(num_classes):
        pred_class = pred_mask == class_idx
        true_class = true_mask == class_idx
        
        intersection = torch.logical_and(pred_class, true_class).sum()
        union = torch.logical_or(pred_class, true_class).sum()
        
        class_iou[class_idx] = intersection / (union + 1e-8)
        
    mean_iou = torch.mean(class_iou)
    return mean_iou, class_iou

if __name__ == "__main__":
    num_classes = 2
    num_samples = 100

    pred_masks = torch.as_tensor(np.random.randint(0, num_classes, size=(num_samples, 256, 256)))
    true_masks = torch.as_tensor(np.random.randint(0, num_classes, size=(num_samples, 256, 256)))

    print(pred_masks.unique())
    # Calculate mIoU
    total_iou = 0.0
    for i in range(num_samples):
        miou, _ = miou(pred_masks[i], true_masks[i], num_classes)
        total_iou += miou

    mean_iou = total_iou / num_samples
    print(f"Mean IoU: {mean_iou}")