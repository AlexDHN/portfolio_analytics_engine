def fees(x: float) -> float:
    """
    Returns the transaction fee charged by Bourse Direct for a PEA active_group_posount.

    The fee depends on the transaction amount based on predefined brackets:
    - Up to €500: €0.99
    - €501 to €1000: €1.90
    - €1001 to €2000: €2.90
    - €2001 to €4400: €3.80
    - Above €4400: 0.09% of the amount

    Args:
        x (float): The amount of the transaction (buy or sell)

    Returns:
        float: The corresponding fee in euros

    Examples:
        >>> fees(500)
        0.99
        >>> fees(1000)
        1.9
        >>> fees(2500)
        3.8
        >>> fees(6000)
        5.4
    """
    if x <= 500:
        return 0.99
    elif x <= 1000:
        return 1.9
    elif x <= 2000:
        return 2.9
    elif x <= 4400:
        return 3.80
    else:
        return x * 0.0009
