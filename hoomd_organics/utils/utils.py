'''utils.py
   utility methods for hoomd-organics
'''

def check_return_iterable(obj):
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, str):
        return [obj]
    try:
        iter(obj)
        return obj
    except:  # noqa: E722
        return [obj]


def scale_charges(charges):
    net_charge = sum(charges)
    abs_charge = sum([abs(charge) for charge in charges])
    scaled_charges = []
    for idx, charge in enumerate(charges):
        new_charge = charge - (abs(charge) * (net_charge / abs_charge))
        scaled_charges.append(new_charge)
    return scaled_charges
