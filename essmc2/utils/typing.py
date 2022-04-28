# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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
        if value is None:
            continue
        if not isinstance(value, dict):
            return False
        if contains_type and 'type' not in value:
            return False

    return True


def check_seq_of_seq(input_list):
    """ Check input list has typing (list, tuple)[(list, tuple)]

    Args:
        input_list (list): List to check.

    Returns:
        Bool.
    """
    if not isinstance(input_list, (list, tuple)):
        return False
    if len(input_list) == 0:
        return False
    for t in input_list:
        if not isinstance(t, (list, tuple)):
            return False
    return True
