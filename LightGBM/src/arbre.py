from boite import Boite

def propagate_boite_in_tree(tree, boite):
    """
    Propagation d'une boîte dans un arbre LightGBM.
    Args:
        tree (dict): Un arbre complet (élément de 'tree_info').
        boite (Boite): La boîte à propager.
    Returns:
        list of (Boite, None): Les boîtes finales partitionnées.
    """
    node = tree['tree_structure']
    return _recursive_partition_lightgbm(node, boite)

def _recursive_partition_lightgbm(node, boite):
    if boite is None:
        return []

    # Cas feuille
    if 'left_child' not in node and 'right_child' not in node:
        return [(boite, None)]  # on ne garde plus les scores

    feature_index = node['split_feature']
    threshold = node['threshold']

    # LightGBM : gauche = <= threshold
    left_boite, right_boite = boite.split(feature_index, threshold)

    results = []
    if left_boite and Boite.is_valid(left_boite):
        results.extend(_recursive_partition_lightgbm(node['left_child'], left_boite))
    if right_boite and Boite.is_valid(right_boite): 
        results.extend(_recursive_partition_lightgbm(node['right_child'], right_boite))

    return results

def propagate_boites_in_tree(tree, boites, class_id=None):
    """
    Propage un ensemble de boîtes dans un arbre LightGBM, sans modification de logits.
    Args:
        tree (dict): Un arbre complet (élément de 'tree_info').
        boites (list of (Boite, None)): Les boîtes à propager.
    Returns:
        list of (Boite, None): Boîtes finales après partitionnement.
    """
    all_outputs = []
    for boite, _ in boites:
        all_outputs.extend(propagate_boite_in_tree(tree, boite))
    return all_outputs
