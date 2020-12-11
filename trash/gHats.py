def gHatCartpole(estimates):
    # return 100 - estimates
    return 170 - estimates
    # return 185 - estimates


def gHat1Gridworld(estimates):
    # return -130 - estimates
    # return 50 - estimates
    return -130 - estimates


def gHat2Gridworld(estimates):
    return estimates + 1


def gHatGridworldv2(estimates):
    return 30 - estimates


def gHat1Mountaincar(estimates):
    # return 100 - estimates
    # return 150 - estimates
    return 150 - estimates


def gHat2Mountaincar(estimates):
    # return -830 - estimates
    return estimates + 1