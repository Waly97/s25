from boite import Boite
import numpy as np
def propagate_boite_in_tree(tree, boite, node_id=0):
    if boite is None:
        return []

    left_children = tree["left_children"]
    right_children = tree["right_children"]
    split_indices = tree["split_indices"]
    split_conditions = tree["split_conditions"]

    if left_children[node_id] == -1 and right_children[node_id] == -1:
        # 🟢 Ne PAS utiliser base_weights
        return [boite]

    feature_index = split_indices[node_id]
    threshold = split_conditions[node_id]

    left_boite, right_boite = boite.split(feature_index, threshold)

    results = []
    if left_boite and Boite.is_valid(left_boite):
        results += propagate_boite_in_tree(tree, left_boite, left_children[node_id])
    if right_boite and Boite.is_valid(right_boite):
        results += propagate_boite_in_tree(tree, right_boite, right_children[node_id])

    return results




def propagate_boites_in_tree(tree, boites, class_id):
    """
    Propagation des boîtes uniquement avec partitionnement (sans accumulation de leaf_values).
    """
    all_outputs = []
    for boite, logits in boites:
        for b in propagate_boite_in_tree(tree, boite):
            new_logits = np.copy(logits)
            all_outputs.append((b, new_logits))
    return all_outputs
