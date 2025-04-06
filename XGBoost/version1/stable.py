from tqdm import tqdm
import boxes as bx

def is_stable(box1, box2):
    for b in box1:
        fmin = b["f_min"]
        fmax = b["f_max"]
        for b1 in box2:
            instance = b1["instances"]
            for i in instance:
                if bx.leq(fmin, i) and bx.leq(i, fmax):
                    return False
    return True

def is_stable_bw_bxs(boxes):
    class_box = [x for x, y in boxes]
    total = len(class_box) * (len(class_box) - 1)

    pbar = tqdm(total=total, desc="Checking stability", ncols=80)
    for i in range(len(class_box)):
        for j in range(len(class_box)):
            if i != j:
                if is_stable(class_box[i], class_box[j]):
                    pbar.update(1)
                    continue
                pbar.update(1)
                pbar.close()
                return False
    pbar.close()
    return True

def verif_stable(boxes):
    if is_stable_bw_bxs(boxes):
        print("✅ The stability has been respected")
    else:
        print("❌ The model isn't stable")
