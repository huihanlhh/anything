from transformers import FlaxLukeModel, LukeModel

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