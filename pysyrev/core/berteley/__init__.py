# This submodule is derived from the berteley package:
#   Authors : Eric Chagnon, Ronald J. Pandolfi, Daniela Ushizima
#             Lawrence Berkeley National Laboratory
#   Source  : https://github.com/lbl-camera/berteley
#   License : BSD
#
# Only the functions used by pysyrev (preprocess, fit, _calculate_metrics)
# have been retained. alive-progress and joblib dependencies have been
# removed; progress reporting is delegated to tqdm, consistent with the
# rest of pysyrev.
