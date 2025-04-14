from boite import Boite
import numpy as np

def propagate_boite_in_tree(tree, boite, node_id=0):
    # Cas où on atteint une feuille
    if tree["left_children"][node_id] == -1 and tree["right_children"][node_id] == -1:
        score = tree["base_weights"][node_id]
        return [(boite, score)]

    feature_index = tree["split_indices"][node_id]
    threshold = tree["split_conditions"][node_id]
    
    left_boite, right_boite = boite.split(feature_index, threshold)
    results = []

    if left_boite is not None:
        left_id = tree["left_children"][node_id]
        results+=(propagate_boite_in_tree(tree, left_boite, left_id))

    if right_boite is not None:
        right_id = tree["right_children"][node_id]
        results+=(propagate_boite_in_tree(tree, right_boite, right_id))

    return results

def propagate_boites_in_tree(arbre_json, boites, class_id):
    """
    Version batch : Propagation d'une liste de (boite, logits) dans un arbre donné.
    Retourne une liste de (boite_finale, logits_mis_a_jour).
    """
    all_outputs = []
    for boite, logits in boites:
        results = propagate_boite_in_tree(arbre_json, boite)
        for b, score in results:
            new_logits = np.copy(logits)
            new_logits[class_id] += score
            all_outputs.append((b, new_logits))
    return all_outputs
