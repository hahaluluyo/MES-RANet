import os
import time
from datetime import datetime
import multiprocessing
from argparse import ArgumentParser

import yaml
import torch

from tool import set_seed
from tem import tem_train_group, tem_output_group, tem_nms, tem_iou_process, \
    tem_final_result_per_subject, tem_final_result_best, tem_train_and_eval


def bi_loss(output, label):
    weight = torch.empty_like(output)
    c_0 = 0.05  # (label > 0).sum / torch.numel(label)
    c_1 = 1 - c_0
    weight[label > 0] = c_1
    weight[label == 0] = c_0
    loss = torch.nn.functional.binary_cross_entropy(output, label, weight)
    return loss


def create_folder(opt):
    # create folder
    output_path = os.path.join(opt['project_root'], opt['output_dir_name'])
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for subject in subject_list:
        out_subject_path = os.path.join(output_path, subject)
        if not os.path.exists(out_subject_path):
            os.mkdir(out_subject_path)
        subject_tem_out = os.path.join(out_subject_path, 'tem_out')
        if not os.path.exists(subject_tem_out):
            os.mkdir(subject_tem_out)
        subject_tem_nms_path = os.path.join(out_subject_path, 'tem_nms')
        if not os.path.exists(subject_tem_nms_path):
            os.mkdir(subject_tem_nms_path)
        subject_tem_final_result_path = os.path.join(
            out_subject_path, 'sub_tem_final_result')
        if not os.path.exists(subject_tem_final_result_path):
            os.mkdir(subject_tem_final_result_path)


def tem_train_mul_process(subject_group, opt):
    print("tem tem_train_mul_process ------ start: ")
    print("tem_training_lr: ", opt["tem_training_lr"])
    print("tem_weight_decay: ", opt["tem_weight_decay"])
    print("tem_lr_scheduler: ", opt["tem_lr_scheduler"])
    print("tem_apex_gamma: ", opt["tem_apex_gamma"])
    print("tem_apex_alpha: ", opt["tem_apex_alpha"])
    print("tem_action_gamma: ", opt["tem_action_gamma"])
    print("tem_action_alpha: ", opt["tem_action_alpha"])

    process = []
    start_time = datetime.now()
    for subject_list in subject_group:
        p = multiprocessing.Process(target=tem_train_group,
                                    args=(opt, subject_list))#多进程运行
        p.start()
        process.append(p)
        time.sleep(1)
    for p in process:
        p.join()
    delta_time = datetime.now() - start_time
    print("tem tem_train_mul_process ------ sucessed: ")
    print("time: ", delta_time)


def tem_output_mul_process(subject_group, opt):
    print("tem_output_mul_process ------ start: ")
    print("micro_apex_score_threshold: ", opt["micro_apex_score_threshold"])
    print("macro_apex_score_threshold: ", opt["macro_apex_score_threshold"])
    process = []
    start_time = datetime.now()
    for subject_list in subject_group:
        p = multiprocessing.Process(target=tem_output_group,
                                    args=(opt, subject_list))
        p.start()
        p.join()
        process.append(p)
    for p in process:
        p.join()
    delta_time = datetime.now() - start_time
    print("tem_output_mul_process ------ sucessed: ")
    print("time: ", delta_time)


def tem_nms_mul_process(subject_list, opt):
    print("tem_nms ------ start: ")
    print("nms_top_K: ", opt["nms_top_K"])
    process = []
    start_time = datetime.now()
    for subject in subject_list:
        p = multiprocessing.Process(target=tem_nms, args=(opt, subject))
        p.start()
        process.append(p)
    for p in process:
        p.join()
    delta_time = datetime.now() - start_time
    print("tem_nms ------ sucessed: ")
    print("time: ", delta_time)


def tem_iou_mul_process(subject_list, opt):
    print("tem_iou_process ------ start: ")
    process = []
    start_time = datetime.now()
    for subject in subject_list:
        p = multiprocessing.Process(target=tem_iou_process,
                                    args=(opt, subject))
        p.start()
        process.append(p)
    for p in process:
        p.join()
    delta_time = datetime.now() - start_time
    print("tem_iou_process ------ sucessed: ")
    print("time: ", delta_time)


if __name__ == "__main__":
    set_seed(seed=42)

    parser = ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--mode")
    args = parser.parse_args()

    # debug
    args.dataset = "samm"
    args.mode = "tem_iou_mul_process"

    with open("./config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        if args.dataset is not None:
            dataset = args.dataset
        else:
            dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    subject_list = opt['subject_list']

    if args.mode is not None:
        opt["mode"] = args.mode

    create_folder(opt)

    print(f"===================== Dataset is {dataset} =====================")

    if dataset != "cross":
        tmp_work_numbers = 5
        subject_group = []
        if len(subject_list) % tmp_work_numbers == 0:
            len_per_group = int(len(subject_list) // tmp_work_numbers)
            for i in range(tmp_work_numbers):
                subject_group.append(subject_list[
                                     i * len_per_group:
                                     (i + 1) * len_per_group])
        else:
            len_per_group = int(len(subject_list) // tmp_work_numbers) + 1
            last_len = len(subject_list) - len_per_group * (tmp_work_numbers - 1)
            for i in range(tmp_work_numbers - 1):
                subject_group.append(subject_list[
                                     i * len_per_group:
                                     (i + 1) * len_per_group])
            subject_group.append(subject_list[-last_len:])

        if opt["mode"] == "tem_train_mul_process":
            tem_train_mul_process(subject_group, opt)
        elif opt["mode"] == "tem_output_mul_process":
            tem_output_mul_process(subject_group, opt)
        elif opt["mode"] == "tem_nms_mul_process":
            tem_nms_mul_process(subject_list, opt)
        elif opt["mode"] == "tem_iou_mul_process":
            tem_iou_mul_process(subject_list, opt)
        elif opt["mode"] == "tem_final_result":
            print("tem_final_result ------ start: ")
            # tem_final_result(opt, subject_list)
            # smic doesn't have macro label
            if dataset != "smic":
                tem_final_result_per_subject(opt, subject_list, type_idx=1)
            tem_final_result_per_subject(opt, subject_list, type_idx=2)
            tem_final_result_per_subject(opt, subject_list, type_idx=0)
            tem_final_result_best(opt, subject_list, type_idx=0)
            # tem_final_result_best(opt, subject_list, type_idx=1)
            # tem_final_result_best(opt, subject_list, type_idx=2)
            print("tem_final_result ------ successed")
    else:
        tem_train_and_eval(opt)
