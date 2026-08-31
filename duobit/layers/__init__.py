from duobit.layers.duobit_embedding import DuobitEmbedding
from duobit.layers.duobit_linear import DuobitLinear
from duobit.layers.ste_linear import SteQuantLinear

# Every module whose weights persist as codes + group scales. The optimizer and
# the memory accounting dispatch on this tuple rather than on DuobitLinear.
DUOBIT_MODULES = (DuobitLinear, DuobitEmbedding)

__all__ = ["DuobitLinear", "DuobitEmbedding", "SteQuantLinear", "DUOBIT_MODULES"]
