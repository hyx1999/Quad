import pickle
import json
import os
import torch
import transformers
from quad.entry.modules import module_utils
from quad.entry.rotation import (
    rotation_utils,
    pod_utils,
    svd_utils,
    hadamard_utils
)
from quad.entry import (
    utils,
    calib_utils,
)

print(os.environ["HF_HOME"])

def main():
    args = utils.parser_gen()
    
    print("args.pod_rank: {}".format(args.pod_rank))
    
    transformers.set_seed(args.seed)
    model = module_utils.get_model(args.model, args.hf_token)
    model.eval()

    results = calib_utils.calib_model(model, args)
    with open("misc/data/calib_results.pkl", "wb") as f:
        pickle.dump(results, f)

    module_utils.untie_word_embedding(model)
    rotation_utils.fuse_layer_norms(model)
    pod_utils.decompose_model(model, args)

    if args.pod_rank > 0:
        results = calib_utils.calib_model(model, args)
        with open("misc/data/calib_results_after_pod.pkl", "wb") as f:
                pickle.dump(results, f)

    rotation_utils.rotate_model(model, args)
    results = calib_utils.calib_model(model, args)
    
    if args.pod_rank > 0:
        with open("misc/data/calib_results_after_quad.pkl", "wb") as f:
            pickle.dump(results, f)
    else:
        with open("misc/data/calib_results_after_quarot.pkl", "wb") as f:
            pickle.dump(results, f)        


if __name__ == '__main__':
    main()
