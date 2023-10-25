import os 
import json 
import numpy as np 
import math

from typing import Union, List
from tqdm import tqdm
from multiprocessing import Queue, Process

import misc_utils 
from matching import maxWeightMatching

PERSON_CLASSES = ['background', 'person']

class CrowdHumanMetric:
    def __init__(self, 
                 gtpath: str = None, 
                 dtpath: str = None, 
                 metric: Union[str, List[str]] = ['AP', 'MR', 'JI'],
                 body_key = None, 
                 head_key = None,
                 iou_thres: float = 0.5,
                 mode: int = 0,
                 num_ji_process: int = 10) -> None:
        
        
        self.gtpath = gtpath
        self.dtpath = dtpath
        self.metric = metric
        self.body_key = body_key
        self.head_key = head_key
        self.iou_thres = iou_thres
        self.eval_mode = mode
        self.num_ji_process = num_ji_process
    
        self.images = dict()

        self.loadData(gtpath, body_key, head_key, True)
        self.loadData(dtpath, body_key, head_key, False)

        self._ignNum = sum([self.images[i]._ignNum for i in self.images])
        self._gtNum = sum([self.images[i]._gtNum for i in self.images])
        self._imageNum = len(self.images)
        self.scorelist = None

    
    def loadData(self, fpath, body_key=None, head_key=None, if_gt=True):
        assert os.path.isfile(fpath), fpath + " does not exist!"
        with open(fpath, "r") as f:
            lines = f.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]
        if if_gt:
            for record in records:
                self.images[record["ID"]] = Image(self.eval_mode)
                self.images[record["ID"]].load(record, body_key, head_key, PERSON_CLASSES, True)
        else:
            for record in records:
                self.images[record["ID"]].load(record, body_key, head_key, PERSON_CLASSES, False)
                self.images[record["ID"]].clip_all_boader()

    def compare(self, matching=None):
        """
        match the detection results with the groundtruth in the whole database
        """
        assert matching is None or matching == "VOC", matching
        scorelist = list()
        for ID in self.images:
            if matching == "VOC":
                result = self.images[ID].compare_voc(self.iou_thres)
            else:
                result = self.images[ID].compare_caltech(self.iou_thres)
            scorelist.extend(result)
        # In the descending sort of dtbox score.
        scorelist.sort(key=lambda x: x[0][-1], reverse=True)
        self.scorelist = scorelist

    def eval_MR(self, ref="CALTECH_-2"):
        """
        evaluate by Caltech-style log-average miss rate
        ref: str - "CALTECH_-2"/"CALTECH_-4"
        """
        # find greater_than
        def _find_gt(lst, target):
            for idx, item in enumerate(lst):
                if item >= target:
                    return idx
            return len(lst)-1

        assert ref == "CALTECH_-2" or ref == "CALTECH_-4", ref
        if ref == "CALTECH_-2":
            # CALTECH_MRREF_2: anchor points (from 10^-2 to 1) as in P.Dollar's paper
            ref = [0.0100, 0.0178, 0.03160, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.000]
        else:
            # CALTECH_MRREF_4: anchor points (from 10^-4 to 1) as in S.Zhang's paper
            ref = [0.0001, 0.0003, 0.00100, 0.0032, 0.0100, 0.0316, 0.1000, 0.3162, 1.000]

        if self.scorelist is None:
            self.compare()

        tp, fp = 0.0, 0.0
        fppiX, fppiY = list(), list() 
        for i, item in enumerate(self.scorelist):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0

            fn = (self._gtNum - self._ignNum) - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            missrate = 1.0 - recall
            fppi = fp / self._imageNum
            fppiX.append(fppi)
            fppiY.append(missrate)

        score = list()
        for pos in ref:
            argmin = _find_gt(fppiX, pos)
            if argmin >= 0:
                score.append(fppiY[argmin])
        score = np.array(score)
        MR = np.exp(np.log(score).mean())
        return MR, (fppiX, fppiY)

    def eval_AP(self):
        """
        :meth: evaluate by average precision
        """
        # calculate general ap score
        def _calculate_map(recall, precision):
            assert len(recall) == len(precision)
            area = 0
            for i in range(1, len(recall)):
                delta_h = (precision[i-1] + precision[i]) / 2
                delta_w = recall[i] - recall[i-1]
                area += delta_w * delta_h
            return area

        tp, fp = 0.0, 0.0
        rpX, rpY = list(), list() 
        total_det = len(self.scorelist)
        total_gt = self._gtNum - self._ignNum
        total_images = self._imageNum

        fpn = []
        recalln = []
        thr = []
        fppi = []
        for i, item in enumerate(self.scorelist):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0
            fn = total_gt - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            rpX.append(recall)
            rpY.append(precision)
            fpn.append(fp)
            recalln.append(tp)
            thr.append(item[0][-1])
            fppi.append(fp/total_images)

        AP = _calculate_map(rpX, rpY)
        return AP, (rpX, rpY, thr, fpn, recalln, fppi)

    def eval_JI(self):
        """
        :meth: evaluate by JI
        """
        records = misc_utils.load_json_lines(self.dtpath)
        res_line = []
        res_ji = []
        for i in range(10):
            score_thr = 1e-1 * i
            total = len(records)
            stride = math.ceil(total / self.num_ji_process)
            result_queue = Queue(10000)
            results, procs = [], []
            for i in range(self.num_ji_process):
                start = i*stride
                end = np.min([start+stride,total])
                sample_data = records[start:end]
                p = Process(target= self.compute_JI_with_ignore, args=(result_queue, sample_data, score_thr, self.body_key))
                p.start()
                procs.append(p)
            tqdm.monitor_interval = 0
            pbar = tqdm(total=total, leave = False, ascii = True)
            for i in range(total):
                t = result_queue.get()
                results.append(t)
                pbar.update(1)
            for p in procs:
                p.join()
            pbar.close()
            line, mean_ratio = self.gather(results)
            line = 'score_thr:{:.1f}, {}'.format(score_thr, line)
            print(line)
            res_line.append(line)
            res_ji.append(mean_ratio)

        return max(res_ji) 

    def compute_JI_with_ignore(self, result_queue, records, score_thr, target_key):
        for record in records:
            gt_boxes = misc_utils.load_bboxes(record, 'gtboxes', target_key, 'tag')
            gt_boxes[:,2:4] += gt_boxes[:,:2]
            gt_boxes = misc_utils.clip_boundary(gt_boxes, record['height'], record['width'])
            dt_boxes = misc_utils.load_bboxes(record, 'dtboxes', target_key, 'score')
            dt_boxes[:,2:4] += dt_boxes[:,:2]
            dt_boxes = misc_utils.clip_boundary(dt_boxes, record['height'], record['width'])
            keep = dt_boxes[:, -1] > score_thr
            dt_boxes = dt_boxes[keep][:, :-1]

            gt_tag = np.array(gt_boxes[:,-1]!=-1)
            matches = self.compute_matching(dt_boxes, gt_boxes[gt_tag, :4])
            # get the unmatched_indices
            matched_indices = np.array([j for (j,_) in matches])
            unmatched_indices = list(set(np.arange(dt_boxes.shape[0])) - set(matched_indices))
            num_ignore_dt = self.get_ignores(dt_boxes[unmatched_indices], gt_boxes[~gt_tag, :4])
            matched_indices = np.array([j for (_,j) in matches])
            unmatched_indices = list(set(np.arange(gt_boxes[gt_tag].shape[0])) - set(matched_indices))
            num_ignore_gt = self.get_ignores(gt_boxes[gt_tag][unmatched_indices], gt_boxes[~gt_tag, :4])
            # compurte results
            eps = 1e-6
            k = len(matches)
            m = gt_tag.sum() - num_ignore_gt
            n = dt_boxes.shape[0] - num_ignore_dt
            ratio = k / (m + n -k + eps)
            recall = k / (m + eps)
            cover = k / (n + eps)
            noise = 1 - cover
            result_dict = dict(ratio = ratio, recall = recall, cover = cover,
                noise = noise, k = k, m = m, n = n)
            result_queue.put_nowait(result_dict)

    def gather(self, results):
        assert len(results)
        img_num = 0
        for result in results:
            if result['n'] != 0 or result['m'] != 0:
                img_num += 1
        mean_ratio = np.sum([rb['ratio'] for rb in results]) / img_num
        mean_cover = np.sum([rb['cover'] for rb in results]) / img_num
        mean_recall = np.sum([rb['recall'] for rb in results]) / img_num
        mean_noise = 1 - mean_cover
        valids = np.sum([rb['k'] for rb in results])
        total = np.sum([rb['n'] for rb in results])
        gtn = np.sum([rb['m'] for rb in results])

        line = 'mean_ratio:{:.4f}, valids:{}, total:{}, gtn:{}'.format(
            mean_ratio, valids, total, gtn)
        return line, mean_ratio


    def compute_matching(self, dt_boxes, gt_boxes):
        assert dt_boxes.shape[-1] > 3 and gt_boxes.shape[-1] > 3
        if dt_boxes.shape[0] < 1 or gt_boxes.shape[0] < 1:
            return list()
        N, K = dt_boxes.shape[0], gt_boxes.shape[0]
        ious = self.compute_iou_matrix(dt_boxes, gt_boxes)
        rows, cols = np.where(ious > self.iou_thres)
        bipartites = [(i + 1, j + N + 1, ious[i, j]) for (i, j) in zip(rows, cols)]
        mates = maxWeightMatching(bipartites)
        if len(mates) < 1:
            return list()
        rows = np.where(np.array(mates) > -1)[0]
        indices = np.where(rows < N + 1)[0]
        rows = rows[indices]
        cols = np.array([mates[i] for i in rows])
        matches = [(i-1, j - N - 1) for (i, j) in zip(rows, cols)]
        return matches

    # def compute_head_body_matching(self, dt_body, dt_head, gt_body, gt_head, bm_thr):
    #     assert dt_body.shape[-1] > 3 and gt_body.shape[-1] > 3
    #     assert dt_head.shape[-1] > 3 and gt_head.shape[-1] > 3
    #     assert dt_body.shape[0] == dt_head.shape[0]
    #     assert gt_body.shape[0] == gt_head.shape[0]
    #     N, K = dt_body.shape[0], gt_body.shape[0]
    #     ious_body = self.compute_iou_matrix(dt_body, gt_body)
    #     ious_head = self.compute_iou_matrix(dt_head, gt_head)
    #     mask_body = ious_body > bm_thr
    #     mask_head = ious_head > bm_thr
    #     # only keep the both matches detections
    #     mask = np.array(mask_body) & np.array(mask_head)
    #     ious = np.zeros((N, K))
    #     ious[mask] = (ious_body[mask] + ious_head[mask]) / 2
    #     rows, cols = np.where(ious > bm_thr)
    #     bipartites = [(i + 1, j + N + 1, ious[i, j]) for (i, j) in zip(rows, cols)]
    #     mates = maxWeightMatching(bipartites)
    #     if len(mates) < 1:
    #         return list()
    #     rows = np.where(np.array(mates) > -1)[0]
    #     indices = np.where(rows < N + 1)[0]
    #     rows = rows[indices]
    #     cols = np.array([mates[i] for i in rows])
    #     matches = [(i-1, j - N - 1) for (i, j) in zip(rows, cols)]
    #     return matches

    # def compute_multi_head_body_matching(self, dt_body, dt_head_0, dt_head_1, gt_body, gt_head, bm_thr):
    #     assert dt_body.shape[-1] > 3 and gt_body.shape[-1] > 3
    #     assert dt_head_0.shape[-1] > 3 and gt_head.shape[-1] > 3
    #     assert dt_head_1.shape[-1] > 3 and gt_head.shape[-1] > 3
    #     assert dt_body.shape[0] == dt_head_0.shape[0]
    #     assert gt_body.shape[0] == gt_head.shape[0]
    #     N, K = dt_body.shape[0], gt_body.shape[0]
    #     ious_body = self.compute_iou_matrix(dt_body, gt_body)
    #     ious_head_0 = self.compute_iou_matrix(dt_head_0, gt_head)
    #     ious_head_1 = self.compute_iou_matrix(dt_head_1, gt_head)
    #     mask_body = ious_body > bm_thr
    #     mask_head_0 = ious_head_0 > bm_thr
    #     mask_head_1 = ious_head_1 > bm_thr
    #     mask_head = mask_head_0 | mask_head_1
    #     # only keep the both matches detections
    #     mask = np.array(mask_body) & np.array(mask_head)
    #     ious = np.zeros((N, K))
    #     #ious[mask] = (ious_body[mask] + ious_head[mask]) / 2
    #     ious[mask] = ious_body[mask]
    #     rows, cols = np.where(ious > bm_thr)
    #     bipartites = [(i + 1, j + N + 1, ious[i, j]) for (i, j) in zip(rows, cols)]
    #     mates = maxWeightMatching(bipartites)
    #     if len(mates) < 1:
    #         return list()
    #     rows = np.where(np.array(mates) > -1)[0]
    #     indices = np.where(rows < N + 1)[0]
    #     rows = rows[indices]
    #     cols = np.array([mates[i] for i in rows])
    #     matches = [(i-1, j - N - 1) for (i, j) in zip(rows, cols)]
    #     return matches

    # def get_head_body_ignores(self, dt_body, dt_head, gt_body, gt_head, bm_thr):
    #     if gt_body.size:
    #         body_ioas = self.compute_ioa_matrix(dt_body, gt_body)
    #         head_ioas = self.compute_ioa_matrix(dt_head, gt_head)
    #         body_ioas = np.max(body_ioas, axis=1)
    #         head_ioas = np.max(head_ioas, axis=1)
    #         head_rows = np.where(head_ioas > bm_thr)[0]
    #         body_rows = np.where(body_ioas > bm_thr)[0]
    #         rows = set.union(set(head_rows), set(body_rows))
    #         return len(rows)
    #     else:
    #         return 0

    def get_ignores(self, dt_boxes, gt_boxes):
        if gt_boxes.size:
            ioas = self.compute_ioa_matrix(dt_boxes, gt_boxes)
            ioas = np.max(ioas, axis = 1)
            rows = np.where(ioas > self.iou_thres)[0]
            return len(rows)
        else:
            return 0

    def compute_ioa_matrix(self, dboxes: np.ndarray, gboxes: np.ndarray):
        eps = 1e-6
        assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
        N, K = dboxes.shape[0], gboxes.shape[0]
        dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
        gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

        iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
        ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
        inter = np.maximum(0, iw) * np.maximum(0, ih)

        dtarea = np.maximum(dtboxes[:,:,2] - dtboxes[:,:,0], 0) * np.maximum(dtboxes[:,:,3] - dtboxes[:,:,1], 0)
        ioas = inter / (dtarea + eps)
        return ioas

    def compute_iou_matrix(self, dboxes:np.ndarray, gboxes:np.ndarray):
        eps = 1e-6
        assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
        N, K = dboxes.shape[0], gboxes.shape[0]
        dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
        gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

        iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
        ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
        inter = np.maximum(0, iw) * np.maximum(0, ih)

        dtarea = (dtboxes[:,:,2] - dtboxes[:,:,0]) * (dtboxes[:,:,3] - dtboxes[:,:,1])
        gtarea = (gtboxes[:,:,2] - gtboxes[:,:,0]) * (gtboxes[:,:,3] - gtboxes[:,:,1])
        ious = inter / (dtarea + gtarea - inter + eps)
        return ious

class Image(object):
    def __init__(self, mode):
        self.ID = None
        self._width = None
        self._height = None
        self.dtboxes = None
        self.gtboxes = None
        self.eval_mode = mode

        self._ignNum = None
        self._gtNum = None
        self._dtNum = None

    def load(self, record, body_key, head_key, class_names, gtflag):
        """
        :meth: read the object from a dict
        """
        if "ID" in record and self.ID is None:
            self.ID = record['ID']
        if "width" in record and self._width is None:
            self._width = record["width"]
        if "height" in record and self._height is None:
            self._height = record["height"]
        if gtflag:
            self._gtNum = len(record["gtboxes"])
            body_bbox, head_bbox = self.load_gt_boxes(record, 'gtboxes', class_names)
            if self.eval_mode == 0:
                self.gtboxes = body_bbox
                self._ignNum = (body_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 1:
                self.gtboxes = head_bbox
                self._ignNum = (head_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 2:
                gt_tag = np.array([body_bbox[i,-1]!=-1 and head_bbox[i,-1]!=-1 for i in range(len(body_bbox))])
                self._ignNum = (gt_tag == 0).sum()
                self.gtboxes = np.hstack((body_bbox[:, :-1], head_bbox[:, :-1], gt_tag.reshape(-1, 1)))
            else:
                raise Exception('Unknown evaluation mode!')
        if not gtflag:
            self._dtNum = len(record["dtboxes"])
            if self.eval_mode == 0:
                self.dtboxes = self.load_det_boxes(record, 'dtboxes', body_key, 'score')
            elif self.eval_mode == 1:
                self.dtboxes = self.load_det_boxes(record, 'dtboxes', head_key, 'score')
            elif self.eval_mode == 2:
                body_dtboxes = self.load_det_boxes(record, 'dtboxes', body_key)
                head_dtboxes = self.load_det_boxes(record, 'dtboxes', head_key, 'score')
                self.dtboxes = np.hstack((body_dtboxes, head_dtboxes))
            else:
                raise Exception('Unknown evaluation mode!')

    def compare_caltech(self, thres):
        """
        :meth: match the detection results with the groundtruth by Caltech matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        dt_matched = np.zeros(dtboxes.shape[0])
        gt_matched = np.zeros(gtboxes.shape[0])

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        if len(dtboxes):
            overlap_iou = self.box_overlap_opr(dtboxes, gtboxes, True)
            overlap_ioa = self.box_overlap_opr(dtboxes, gtboxes, False)
        else:
            return list()

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres
            for j, gt in enumerate(gtboxes):
                if gt_matched[j] == 1:
                    continue
                if gt[-1] > 0:
                    overlap = overlap_iou[i][j]
                    if overlap > maxiou:
                        maxiou = overlap
                        maxpos = j
                else:
                    if maxpos >= 0:
                        break
                    else:
                        overlap = overlap_ioa[i][j]
                        if overlap > thres:
                            maxiou = overlap
                            maxpos = j
            if maxpos >= 0:
                if gtboxes[maxpos, -1] > 0:
                    gt_matched[maxpos] = 1
                    dt_matched[i] = 1
                    scorelist.append((dt, 1, self.ID))
                else:
                    dt_matched[i] = -1
            else:
                dt_matched[i] = 0
                scorelist.append((dt, 0, self.ID))
        return scorelist

    def compare_caltech_union(self, thres):
        """
        :meth: match the detection results with the groundtruth by Caltech matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        if len(dtboxes) == 0:
            return list()
        dt_matched = np.zeros(dtboxes.shape[0])
        gt_matched = np.zeros(gtboxes.shape[0])

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        dt_body_boxes = np.hstack((dtboxes[:, :4], dtboxes[:, -1][:,None]))
        dt_head_boxes = dtboxes[:, 4:8]
        gt_body_boxes = np.hstack((gtboxes[:, :4], gtboxes[:, -1][:,None]))
        gt_head_boxes = gtboxes[:, 4:8]
        overlap_iou = self.box_overlap_opr(dt_body_boxes, gt_body_boxes, True)
        overlap_head = self.box_overlap_opr(dt_head_boxes, gt_head_boxes, True)
        overlap_ioa = self.box_overlap_opr(dt_body_boxes, gt_body_boxes, False)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres
            for j, gt in enumerate(gtboxes):
                if gt_matched[j] == 1:
                    continue
                if gt[-1] > 0:
                    o_body = overlap_iou[i][j]
                    o_head = overlap_head[i][j]
                    if o_body > maxiou and o_head > maxiou:
                        maxiou = o_body
                        maxpos = j
                else:
                    if maxpos >= 0:
                        break
                    else:
                        o_body = overlap_ioa[i][j]
                        if o_body > thres:
                            maxiou = o_body
                            maxpos = j
            if maxpos >= 0:
                if gtboxes[maxpos, -1] > 0:
                    gt_matched[maxpos] = 1
                    dt_matched[i] = 1
                    scorelist.append((dt, 1, self.ID))
                else:
                    dt_matched[i] = -1
            else:
                dt_matched[i] = 0
                scorelist.append((dt, 0, self.ID))
        return scorelist

    def box_overlap_opr(self, dboxes:np.ndarray, gboxes:np.ndarray, if_iou):
        eps = 1e-6
        assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
        N, K = dboxes.shape[0], gboxes.shape[0]
        dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
        gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

        iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
        ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
        inter = np.maximum(0, iw) * np.maximum(0, ih)

        dtarea = (dtboxes[:,:,2] - dtboxes[:,:,0]) * (dtboxes[:,:,3] - dtboxes[:,:,1])
        if if_iou:
            gtarea = (gtboxes[:,:,2] - gtboxes[:,:,0]) * (gtboxes[:,:,3] - gtboxes[:,:,1]) 
            ious = inter / (dtarea + gtarea - inter + eps)
        else:
            ious = inter / (dtarea + eps)
        return ious

    def clip_all_boader(self):

        def _clip_boundary(boxes,height,width):
            assert boxes.shape[-1]>=4
            boxes[:,0] = np.minimum(np.maximum(boxes[:,0],0), width - 1)
            boxes[:,1] = np.minimum(np.maximum(boxes[:,1],0), height - 1)
            boxes[:,2] = np.maximum(np.minimum(boxes[:,2],width), 0)
            boxes[:,3] = np.maximum(np.minimum(boxes[:,3],height), 0)
            return boxes

        assert self.dtboxes.shape[-1]>=4
        assert self.gtboxes.shape[-1]>=4
        assert self._width is not None and self._height is not None
        if self.eval_mode == 2:
            self.dtboxes[:, :4] = _clip_boundary(self.dtboxes[:, :4], self._height, self._width)
            self.gtboxes[:, :4] = _clip_boundary(self.gtboxes[:, :4], self._height, self._width)
            self.dtboxes[:, 4:8] = _clip_boundary(self.dtboxes[:, 4:8], self._height, self._width)
            self.gtboxes[:, 4:8] = _clip_boundary(self.gtboxes[:, 4:8], self._height, self._width)
        else:
            self.dtboxes = _clip_boundary(self.dtboxes, self._height, self._width)
            self.gtboxes = _clip_boundary(self.gtboxes, self._height, self._width)

    def load_gt_boxes(self, dict_input, key_name, class_names):
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        head_bbox = []
        body_bbox = []
        for rb in dict_input[key_name]:
            if rb['tag'] in class_names:
                body_tag = class_names.index(rb['tag'])
                head_tag = 1
            else:
                body_tag = -1
                head_tag = -1
            if 'extra' in rb:
                if 'ignore' in rb['extra']:
                    if rb['extra']['ignore'] != 0:
                        body_tag = -1
                        head_tag = -1
            if 'head_attr' in rb:
                if 'ignore' in rb['head_attr']:
                    if rb['head_attr']['ignore'] != 0:
                        head_tag = -1
            head_bbox.append(np.hstack((rb['hbox'], head_tag)))
            body_bbox.append(np.hstack((rb['fbox'], body_tag)))
        head_bbox = np.array(head_bbox)
        head_bbox[:, 2:4] += head_bbox[:, :2]
        body_bbox = np.array(body_bbox)
        body_bbox[:, 2:4] += body_bbox[:, :2]
        return body_bbox, head_bbox

    def load_det_boxes(self, dict_input, key_name, key_box, key_score=None, key_tag=None):
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        else:
            assert key_box in dict_input[key_name][0]
            if key_score:
                assert key_score in dict_input[key_name][0]
            if key_tag:
                assert key_tag in dict_input[key_name][0]
        if key_score:
            if key_tag:
                bboxes = np.vstack([np.hstack((rb[key_box], rb[key_score], rb[key_tag])) for rb in dict_input[key_name]])
            else:
                bboxes = np.vstack([np.hstack((rb[key_box], rb[key_score])) for rb in dict_input[key_name]])
        else:
            if key_tag:
                bboxes = np.vstack([np.hstack((rb[key_box], rb[key_tag])) for rb in dict_input[key_name]])
            else:
                bboxes = np.vstack([rb[key_box] for rb in dict_input[key_name]])
        bboxes[:, 2:4] += bboxes[:, :2]
        return bboxes

    def compare_voc(self, thres):
        """
        :meth: match the detection results with the groundtruth by VOC matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        if self.dtboxes is None:
            return list()
        dtboxes = self.dtboxes
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        dtboxes.sort(key=lambda x: x.score, reverse=True)
        gtboxes.sort(key=lambda x: x.ign)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres

            for j, gt in enumerate(gtboxes):
                overlap = dt.iou(gt)
                if overlap > maxiou:
                    maxiou = overlap
                    maxpos = j

            if maxpos >= 0:
                if gtboxes[maxpos].ign == 0:
                    gtboxes[maxpos].matched = 1
                    dtboxes[i].matched = 1
                    scorelist.append((dt, self.ID))
                else:
                    dtboxes[i].matched = -1
            else:
                dtboxes[i].matched = 0
                scorelist.append((dt, self.ID))
        return scorelist