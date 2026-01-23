#!/bin/bash
export VK_LAYER_PATH=/opt/homebrew/opt/vulkan-validationlayers/share/vulkan/explicit_layer.d
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
cd build && cmake .. && make && cd .. && ./build/VulkanImagePlayer
# ./build/VulkanImagePlayer
