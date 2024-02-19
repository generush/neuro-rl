from enum import Enum

class NormalizationType(Enum):
    Z_SCORE = 'z_score_normalize'
    RANGE = 'range_normalize'
    RANGE_SOFT = 'range_soft_normalize'