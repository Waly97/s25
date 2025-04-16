from boite import Boite
import numpy as np

def propagate_boite_in_tree(tree, boite, node_id=0):
    if boite is None:
        return []

    # Si feuille → on retourne le score
    if tree["left_children"][node_id] == -1 and tree["right_children"][node_id] == -1:
        return [(boite, tree["base_weights"][node_id])]

    feature_index = tree["split_indices"][node_id]
    threshold = tree["split_conditions"][node_id]

    left_boite, right_boite = boite.split(feature_index, threshold)
    results = []

    if left_boite:
        results.extend(propagate_boite_in_tree(tree, left_boite, tree["left_children"][node_id]))

    if right_boite:
        results.extend(propagate_boite_in_tree(tree, right_boite, tree["right_children"][node_id]))

    return results

def propagate_boites_in_tree(arbre_json, boites, class_id):
    all_outputs = []
    for boite, logits in boites:
        for b, score in propagate_boite_in_tree(arbre_json, boite):
            new_logits = np.copy(logits)
            new_logits[class_id] += score
            all_outputs.append((b, new_logits))
    return all_outputs
