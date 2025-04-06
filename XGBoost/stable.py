from tqdm import tqdm
import boxes as bx  # ton module contenant leq, f_min, f_max (si besoin)

def is_stable(box1, box2):
    for b in box1:
        fmin = b["f_min"]
        fmax = b["f_max"]
        for b1 in box2:
            for i in b1["instances"]:
                if bx.leq(fmin, i) and bx.leq(i, fmax):
                    return False
    return True

def is_stable_bw_bxs(boxes_by_class):
    class_ids = list(boxes_by_class.keys())
    total = len(class_ids) * (len(class_ids) - 1)

    pbar = tqdm(total=total, desc="Checking stability", ncols=80)
    for i in range(len(class_ids)):
        for j in range(len(class_ids)):
            if i != j:
                bxs_i = boxes_by_class[class_ids[i]]
                bxs_j = boxes_by_class[class_ids[j]]
                if is_stable(bxs_i, bxs_j):
                    pbar.update(1)
                    continue
                pbar.update(1)
                pbar.close()
                return False
    pbar.close()
    return True

def verif_stable(boxes_by_class):
    if is_stable_bw_bxs(boxes_by_class):
        print("The stability has been respected")
    else:
        print("The model isn't stable")
