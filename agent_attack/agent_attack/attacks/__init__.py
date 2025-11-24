from .bim import bim
from .clip_attack import ClipAttack
from .pgd import pgd


def get_attack_fn(attack, *args):
    if attack == "pgd":
        return pgd
    elif attack == "bim":
        return bim
    elif attack == "clip_attack":
        return ClipAttack
    else:
        raise ValueError(f"Attack {attack} not supported.")
