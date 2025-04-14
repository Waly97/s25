from tqdm import tqdm
from boite import Boite  # ton module contenant leq, f_min, f_max (si besoin)
from multiprocessing import Pool , cpu_count
from itertools import combinations
from arbre import  propagate_boite_in_tree
from build_boite import  BoitePropagator



class StabilityChecker:

    def __init__(self,boxes_by_class,model):
        self.boxes_by_class = boxes_by_class
        self.model=model

    def leq(self,i1,i2):
        return all(i1[f]<=i2[f] for f in i1)
    
    def is_valid(self,boxe1,boxe2):
        b1= Boite.to_interval_instance(boxe1)
        b2=Boite.to_interval_instance(boxe2)
        if (self.leq(b1["fmax"],b2["fmin"])):
            return Boite.from_bounds(b1["fmax"],b2["fmin"])
        if (self.leq(b2["fmax"],b1["fmin"])):
            return Boite.from_bounds(b2["fmax"],b1["fmin"])
        return None
        
    def generate_inter_boxes_class(self,boxes):
        inter_boxes =[]

        for b1,b2 in combinations(boxes,2):
            inter = self.is_valid(b1,b2)
            if inter is not None:
                inter_boxes.append(inter)
        return inter_boxes


    def _is_stable_intra_class(self,class_id,boxes):
        min_boxes,max_boxes= self.extract_minmax_boxes(boxes)
        inter_boxes=self.generate_inter_boxe_ameliorer(min_boxes,max_boxes)

        for b in inter_boxes:
            result = BoitePropagator(model_json_path=self.model,boite_init=b).run()
            if not all(b["prediction"]== class_id for b in result ):
                return False
        return True
    

    def _verif_stable_intra_class(self):

        for class_id,boxes in self.boxes_by_class.items():
            if not self._is_stable_intra_class(class_id,boxes):
                print("stability has broken")
                return False
        print("Intra_class stability is respected")
        return True

    def is_minimal(self,instance,boxe):
        return not any(self.leq(other,instance) and other != instance for other in boxe)
    
    def is_maximal(self,instance,boxe):
        return not any(self.leq(instance,other) and other != instance for other in boxe)
    
    def extract_minmax_boxes(self,boxes):
        fmins = [Boite.f_min(b) for b in boxes]
        fmaxs =[Boite.f_max(b) for b in boxes]

        min_boxes = [b for b in boxes if self.is_minimal(Boite.f_min(b),fmins)]
        max_boxes =[b for b in boxes if self.is_maximal(Boite.f_max(b),fmaxs)]
        return min_boxes,max_boxes
    
    def generate_inter_boxe_ameliorer(self,min_boxes,max_boxes):
        inter_boxes= []
        total = len(min_boxes) * len(max_boxes)
        pbar = tqdm(total=total, desc="create Boxe inter",ncols=80)
        nb_boxes =0
        for bmin in min_boxes:
            for bmax in max_boxes:
                pbar.update(1)
                i=Boite.from_bounds(Boite.f_min(bmin),Boite.f_max(bmax))
                inter_boxes.append(i)
                nb_boxes += 1
        print("nombre de boxes intermediare ", nb_boxes)        
        pbar.close()
        return inter_boxes
    


    def is_stable(self,box1, box2):
        fmin1, fmax1 = box1["fmin"],box1["fmax"]
    
        fmin2 ,fmax2=  box2["fmin"],box2["fmax"]

        if self.leq(fmin1,fmin2):
            if self.leq(fmin2,fmax1) or self.leq(fmax2,fmax1):
                return False
        return True

    def is_stable_list(self,boxes1,boxes2):
        total= len(boxes1) * len(boxes2)
        pbar = tqdm(total=total, desc="statability boxes list",ncols=80)
        for b in boxes1:
            i1= Boite.to_interval_instance(b)
            for b1 in boxes2:
                i2 = Boite.to_interval_instance(b1)
                if (not self.is_stable(i1,i2)) or (not self.is_stable(i2,i1)):
                    pbar.update(1)
                    print("instance1",i1)
                    print("instance2", i2)
                    pbar.close()
                    return False
                pbar.update(1)
        pbar.close()
        return True

    def check_pair_stability(self,args):
        bxs1,bxs2 =args
        return self.is_stable_list(bxs1,bxs2)

    def is_stable_boxes_by_class(self,boxes_by_class):
        keys = list(boxes_by_class.keys())  # pour garder l’ordre
        total = len(keys) * (len(keys) - 1) // 2
        pbar = tqdm(total=total, desc="Checking stability", ncols=80)

        # On fait une copie pour pouvoir supprimer
        boxes_copy = boxes_by_class.copy()

        while len(boxes_copy) > 1:
            c, bxs1 = next(iter(boxes_copy.items()))
            boxes_copy.pop(c)

            for c2, bxs2 in boxes_copy.items():
                if not self.is_stable_list(bxs1, bxs2):
                    pbar.update(1)
                    pbar.close()
                    return False
                pbar.update(1)

        pbar.close()
        return True

    def is_stable_parallele(self):
        class_labels = list(self.boxes_by_class.keys())
        tasks =[]

        for i in range(len(class_labels)):
            for j in range(i+1,len(class_labels)):
                tasks.append((self.boxes_by_class[class_labels[i]],self.boxes_by_class[class_labels[j]]))
        num_workers = 3
        with Pool(num_workers) as pool:
            result = list(tqdm(pool.imap(self.check_pair_stability,tasks),total=len(tasks),desc="Checking stability",ncols=80))

        return all(result)
    

    def verif_stable(self):
        # if self.is_stable_parallele():
        if self._verif_stable_intra_class():
                print("The stability has been respected")
        else:
            print("The model isn't stable")
