import json
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
import re

def is_valide_answer_format(answer):
    pattern = r'^\{\<\d+\>\<\d+\>\<\d+\>\<\d+\>\|\<\d+\>\}$'
    return re.match(pattern, answer) is not None

# 将标准框转为多边形
def standard_box_to_polygon(bbox):
    xmin, ymin, xmax, ymax = bbox
    return Polygon([
        (xmin, ymin),  # 左下
        (xmax, ymin),  # 右下
        (xmax, ymax),  # 右上
        (xmin, ymax)   # 左上
    ])

# 将旋转框转为多边形
def rotated_box_to_polygon(center, width, height, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cx, cy = center

    # 计算矩形的四个顶点（未旋转）
    corners = np.array([
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2]
    ])

    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])

    # 应用旋转矩阵，并平移到中心点
    rotated_corners = np.dot(corners, rotation_matrix) + [cx, cy]

    return Polygon(rotated_corners)

def transform_prebox(xmin, ymin, xmax, ymax, scale_ratio):
    xmin = (xmin / 100)*scale_ratio
    ymin = (ymin / 100)*scale_ratio
    xmax = (xmax / 100)*scale_ratio
    ymax = (ymax / 100)*scale_ratio
    return int(xmin), int(ymin), int(xmax), int(ymax)

# 计算旋转框和标准框之间的 IoU
def calculate_rotated_iou(pred_bbox, pred_angle, gt_bbox):
    # 解构预测框和真实框
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_bbox
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = transform_prebox(pred_xmin, pred_ymin, pred_xmax, pred_ymax, scale_ratio=5.0*1.587)

    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox

    # 获取旋转框多边形
    pred_center = ((pred_xmin + pred_xmax) / 2, (pred_ymin + pred_ymax) / 2)
    pred_width = pred_xmax - pred_xmin
    pred_height = pred_ymax - pred_ymin
    pred_polygon = rotated_box_to_polygon(pred_center, pred_width, pred_height, pred_angle)

    # 获取标准框多边形
    gt_polygon = standard_box_to_polygon([gt_xmin, gt_ymin, gt_xmax, gt_ymax])

    # 计算交集和并集
    intersection_area = pred_polygon.intersection(gt_polygon).area
    union_area = pred_polygon.union(gt_polygon).area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

# 从 answer 中解析预测框和旋转角度
def parse_answer(answer):
    answer = answer.strip()
    if is_valide_answer_format(answer):
        bbox_part, angle_part = answer.split('|')[0], answer.split('|')[1]
        bbox_values = [int(v.strip('<>')) for v in bbox_part.split('><')]
        angle = int(angle_part.strip('<>'))
        # 返回 (xmin, ymin, xmax, ymax), angle
        return bbox_values, angle  
    else:
        return None, None

# 评估 JSONL 数据中的预测结果
def evaluate_jsonl(jsonl_file):
    iou_scores = []
    total_test, invalid_test, valid_test = 0, 0, 0
    with open(jsonl_file, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            total_test += 1
            # 解析预测框
            pred_bbox, pred_angle = parse_answer(data['answer'])
            if pred_bbox == None and pred_angle == None:
                invalid_test += 1
                continue
            valid_test += 1
            # 解析真实框
            gt_bbox = [
                data['ground_truth']['xmin'],
                data['ground_truth']['ymin'],
                data['ground_truth']['xmax'],
                data['ground_truth']['ymax']
            ]

            # 计算 IoU
            iou = calculate_rotated_iou(pred_bbox, pred_angle, gt_bbox)
            iou_scores.append(iou)

    # 计算平均 IoU
    print("total_test:{}, invalid_test:{}, valid_test:{}".format(total_test, invalid_test, valid_test))
    avg_iou = np.mean(iou_scores) if iou_scores else 0.0
    return avg_iou

# 示例用法
if __name__ == "__main__":
    # 替换为你的 JSONL 文件路径
    jsonl_file = "./answer_file/answer.jsonl"

    avg_iou = evaluate_jsonl(jsonl_file)
    print(f"Average IoU (with rotation): {avg_iou:.4f}")