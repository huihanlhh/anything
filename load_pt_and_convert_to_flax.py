from transformers import FlaxLukeModel, LukeModel
from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
import jax.numpy as jnp

# Load and save the pt model
pt_model = LukeModel.from_pretrained("studio-ousia/luke-base", cache_dir="~/.cache/huggingface/transformers/luke")
pt_model.save_pretrained("~/.cache/huggingface/transformers/luke")

# Load the flax model from the same directory to use the config and model weights
# The `from_pt` argument should take care of weight conversion
flax_model = FlaxLukeModel.from_pretrained("~/.cache/huggingface/transformers/luke", from_pt=True)
flax_model.save_pretrained("~/.cache/huggingface/transformers/luke")

# IMPORTANT!!!
# This code works great for LUKE.
# For mLUKE, we might need to make some changes to `modeling_flax_luke.py`,
# because mLUKE checkpoint does not have entity-aware attention. Here we might need to do some weight renaming or other tricks.

# Another useful function is the following (referring to https://github.com/huggingface/transformers/blob/c55d6e4e10ce2d9c37e5f677f0842b04ef8b73f3/tests/test_modeling_flax_common.py#L271"
# This might be helpful if we end up having to convert by hand
pt_model = LukeModel.from_pretrained("studio-ousia/luke-base", cache_dir="~/.cache/huggingface/transformers/luke")
fx_model = FlaxLukeModel(pt_model.config, dtype=jnp.float32)
fx_state = convert_pytorch_state_dict_to_flax(pt_model.state_dict(), fx_model)
fx_model.params = fx_state