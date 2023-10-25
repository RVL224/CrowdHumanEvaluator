import argparse
import logging
from collections import OrderedDict

from crowdhuman_metric import CrowdHumanMetric

# gt_path = '/home/rvl224/文件/PaperCode/SW-YOLOX-main/datasets/crowdhuman/annotation_val.odgt'
dbName = 'human'

def compute_APMRJI(gt_path, dt_path, metrics=['AP', 'MR', 'JI'], target_key='box', mode=0):
    CHM = CrowdHumanMetric(gtpath = gt_path,
                           dtpath = dt_path, 
                           metric = metrics, 
                           body_key = target_key, 
                           head_key = None,
                           iou_thres = 0.4, 
                           mode=mode)

    print("Compare gtbbox and dtbbox...")
    CHM.compare()
    print("Done.")

    eval_results = {}
    for metric in metrics:
        print(f'Evaluating {metric}...')
        if metric == 'AP':
            mAP,_ = CHM.eval_AP()
            eval_results['mAP'] = float(f'{round(mAP, 4)}')
            print("mAP:{:.4f}".format(mAP))
        if metric == 'MR':
            mMR,_ = CHM.eval_MR()
            eval_results['mMR'] = float(f'{round(mMR, 4)}')
            print("mMR:{:.4f}".format(mMR))
        if metric == 'JI':
            mJI = CHM.eval_JI()
            eval_results['JI'] = float(f'{round(mJI, 4)}')
            print("mJI:{:.4f}".format(mJI))

    line = 'mAP:{:.4f}, mMR:{:.4f}, mJI:{:.4f}'.format(mAP, mMR, mJI)
    print(line)
    print(eval_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze a json result file with iou match')
    parser.add_argument('--gtfile', 
                        default='/home/rvl224/文件/PaperCode/SW-YOLOX-main/datasets/crowdhuman/annotation_val.odgt',
                        required=False, 
                        help='path of groundtruth result file to load')
    parser.add_argument('--detfile', 
                        required=True, 
                        help='path of json result file to load')
    parser.add_argument('--target_key', 
                        default='box', 
                        required=False)
    args = parser.parse_args()
    print(args)

    compute_APMRJI(gt_path=args.gtfile, dt_path=args.detfile, target_key=args.target_key, mode=0)
