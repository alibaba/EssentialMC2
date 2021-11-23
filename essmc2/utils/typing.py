def check_dict_of_str_dict(input_dict, contains_type=False):
    """ Check input dict has typing Dict[str, dict]

    Args:
        input_dict (dict): Dict to check.
        contains_type (bool): Check if sub dict contains key 'type'.

    Returns:
        Bool.
    """
    if not isinstance(input_dict, dict):
        return False
    for key, value in input_dict.items():
        if not isinstance(key, str):
            return False
        if not isinstance(value, dict):
            return False
        if contains_type and 'type' not in value:
            return False

    return True
