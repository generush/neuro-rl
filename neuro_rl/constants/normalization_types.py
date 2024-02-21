from enum import Enum

class NormalizationType(Enum):
    Z_SCORE = 'Z_SCORE' # z_score_normalize
    RANGE = 'RANGE' # range_normalize
    RANGE_SOFT = 'RANGE_SOFT' # range_soft_normalize