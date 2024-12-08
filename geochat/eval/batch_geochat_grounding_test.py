import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import xml.etree.ElementTree as ET
import random
import cv2
import numpy as np

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    # 提取文件名
    filename = root.find('filename').text

    # 提取图片尺寸
    size = root.find('size')
    width = size.find('width').text
    height = size.find('height').text
    depth = size.find('depth').text

    # 提取对象信息
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        pose = obj.find('pose').text
        description = obj.find('description').text if obj.find('description') is not None else None

        # 提取边界框
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text

        # 将对象信息存储为字典
        objects.append({
            'name': name,
            'pose': pose,
            'description': description,
            'bndbox': {
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax)
            }
        })

    return {
        'filename': filename,
        'size': {'width': int(width), 'height': int(height), 'depth': int(depth)},
        'objects': objects
    }

def Read_TFile(tfile_path, ann_path, data_path, dataset='dior_rsvg'):
    # read test files for dataset
    with open(tfile_path, 'r') as file:
        lines = file.readlines()
    
    ann_names = sorted(os.listdir(os.path.join(data_path, ann_path)))
    test_nums = [int(line.strip()) for line in lines]

    xml_names = [ann_names[i] for i in test_nums if i < len(ann_names)]

    return xml_names

def Test_Aug(image, annotation, aug_img_names, crop_size=(800, 800)):
    # image: PIL.Image.
    # annotation: the annotation for current image.
    # aug_path: the natural images path to augment the rs image.
    # the augment image is from coco, (460, 640, 3)
    images = []
    # 随机读取四个图像
    while len(images) < 4:
        aug_img_name = random.sample(aug_img_names, 1)
        image = cv2.imread(aug_img_name[0])
        if image is not None:
            image = cv2.resize(image, (400, 400))
            images.append(image)

    random.shuffle(images)
    h, w, _ = images[0].shape
    # 拼接图像
    top_row = np.hstack((images[0], images[1]))
    bot_row = np.hstack((images[2], images[3]))
    combined_image = np.vstack((top_row, bot_row))
    # 获取拼接后的中心并进行中心剪裁
    combined_h, combined_w, _ = combined_image.shape
    crop_h, crop_w = crop_size
    start_h = (combined_h - crop_h) // 2
    start_w = (combined_w - crop_w) // 2
    cropped_image = combined_image[start_h:start_h+crop_h, start_w:start_w+crop_w]
    # 把遥感图像的grounding区域直接移动到拼接后的图像上来
    object_location = annotation['objects'][0]['bndbox']
    xmin, ymin, xmax, ymax = object_location["xmin"], object_location["ymin"], object_location["xmax"], object_location["ymax"]
    # print("xmin:{}, ymin:{}, xmax:{}, ymax:{}".format(xmin, ymin, xmax, ymax))
    image = np.array(image)
    print(image.shape)
    cropped_image[ymin:ymax, xmin:xmax] = image[ymin:ymax, xmin:xmax]
    return cropped_image


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # read annotation files for DIOR-RSVG
    ann_names = Read_TFile(args.tfile_path, args.annotation_path, args.data_path)    
    anns = []
    for name in ann_names:
        ann_path = os.path.join(os.path.join(args.data_path, args.annotation_path), name)
        ann_info = parse_xml(ann_path)
        anns.append(ann_info)
    for i in tqdm(range(0, len(anns), args.batch_size)):
        input_batch = []
        count = i
        image_folder = []
        batch_end = min(i + args.batch_size, len(anns))

        for j in range(i, batch_end):
            image_file = anns[j]['filename']

        
            qs = "Give bounding box for " + anns[j]['objects'][0]['description'].lower() + "."
            # print(qs)

            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs   

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            input_batch.append(input_ids)

            image = Image.open(os.path.join(args.image_folder, image_file))

            if args.test_mode == "aug":
                aug_img_names = os.listdir(args.aug_path)
                aug_img_names = [os.path.join(args.aug_path, aug_img_name) for aug_img_name in aug_img_names]
                image = Test_Aug(image, anns[j], aug_img_names)

            image_folder.append(image)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        max_length = max(tensor.size(1) for tensor in input_batch)

        final_input_list = [torch.cat((torch.zeros((1,max_length - tensor.size(1)), dtype=tensor.dtype,device=tensor.get_device()), tensor),dim=1) for tensor in input_batch]
        final_input_tensors=torch.cat(final_input_list,dim=0)
        image_tensor_batch = image_processor.preprocess(image_folder,crop_size ={'height': 504, 'width': 504},size = {'shortest_edge': 504}, return_tensors='pt')['pixel_values']
        with torch.inference_mode():
            output_ids = model.generate( final_input_tensors, images=image_tensor_batch.half().cuda(), do_sample=False , temperature=args.temperature, top_p=args.top_p, num_beams=1, max_new_tokens=256,length_penalty=2.0, use_cache=True)

        input_token_len = final_input_tensors.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        for k in range(0,len(final_input_list)):
            output = outputs[k].strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()

            ans_id = shortuuid.uuid()
            
            ans_file.write(json.dumps({
                                    "image_id": anns[i+k]['filename'],
                                    "answer": output,
                                    "ground_truth": anns[i+k]['objects'][0]['bndbox'],
                                    "question":anns[i+k]['objects'][0]['description'],                          
                                    }) + "\n")
            count=count+1
            ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data1/gyl/RS_Code/geochat-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="/data1/gyl/RS_DATASET/DIOR-RSVG")
    parser.add_argument("--tfile-path", type=str, default="/data1/gyl/RS_DATASET/DIOR-RSVG/test.txt")
    parser.add_argument("--annotation-path", type=str, default="Annotations/")
    parser.add_argument("--image-folder", type=str, default="/data1/gyl/RS_DATASET/DIOR-RSVG/JPEGImages")
    parser.add_argument("--aug-path", type=str, default="/data1/slg/dataset/coco/train2017")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="./answer_file/aug_answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size",type=int, default=1)
    parser.add_argument("--test_mode",type=str, default="aug")
    args = parser.parse_args()

    eval_model(args)