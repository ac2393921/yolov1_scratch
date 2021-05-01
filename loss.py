import torch
import torch.nn as nn

from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()

        self.mse = nn.MSELoss(reduction="sum")
        self.S = S  # S is split size of image (in paper 7)
        self.B = B  # B is number of boxes (in paper 2)
        self.C = C  # C is number of classes (in paper and VOC dataset is 20)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)

        # Calculate the IoU for each
        # それぞれのIoUを計算
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # 2つの予測値のうち、IoUが最も高い矩形を取る。
        iou_maxes, best_box = torch.max(ious, dim=0)  # best_box is index of maxes IoU.
        exists_box = target[..., 20].unsqueeze(3)  # identity_obj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        # Loss of box center coordinates and box size
        # ボックスの中心座標とボックスの大きさの損失
        box_predictions = exists_box * (
            (
                best_box * predictions[..., 26:30]
                + (1 - best_box) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        # (N, S, S, 25)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        # Measuring Trust. The confidence score for the bbox with highest IoU
        # 信頼度を測定　IoUが最も高いbboxの信頼度スコア
        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )

        # (N*S*S, 1)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        # Unreliability
        # Calculate the largest non-IoU
        # 不信頼度であり、最大のIoUでないものを計算
        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        # Loss of class confidence
        # クラス確信度の損失
        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=2),
            torch.flatten(exists_box * target[..., :20], end_dim=2)
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
