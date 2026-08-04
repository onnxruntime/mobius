# image_processor.json — Complete Field Reference

This document is the exhaustive reference for the `image_processor.json` file
used by ort-extensions image processors. For an overview, see the parent
[SKILL.md](../SKILL.md).

---

## Format: ort-extensions vs HuggingFace

> **Critical:** ORT GenAI expects the ort-extensions format — NOT the
> HuggingFace format (HF's `preprocessor_config.json` wraps data under
> `"image_processor"` with different keys). The ort-extensions
> `image_processor.json` expects a `"processor"` key with an ordered transform pipeline.

---

## Qwen2.5-VL full example

```json
{
  "processor": {
    "name": "qwen2_5_image_processor",
    "transforms": [
      {
        "operation": {
          "name": "decode_image",
          "type": "DecodeImage",
          "attrs": { "color_space": "RGB" }
        }
      },
      {
        "operation": {
          "name": "convert_to_rgb",
          "type": "ConvertRGB"
        }
      },
      {
        "operation": {
          "name": "resize",
          "type": "Resize",
          "attrs": {
            "width": 540,
            "height": 360,
            "smart_resize": 1,
            "min_pixels": 3136,
            "max_pixels": 12845056,
            "patch_size": 14,
            "merge_size": 2
          }
        }
      },
      {
        "operation": {
          "name": "rescale",
          "type": "Rescale",
          "attrs": { "rescale_factor": 0.00392156862745098 }
        }
      },
      {
        "operation": {
          "name": "normalize",
          "type": "Normalize",
          "attrs": {
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
            "qwen2_5_vl": 1
          }
        }
      },
      {
        "operation": {
          "name": "patch_image",
          "type": "PatchImage",
          "attrs": {
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2
          }
        }
      }
    ]
  }
}
```

---

## Transform types

| Type | Purpose | Key attrs |
|---|---|---|
| `DecodeImage` | Decode from bytes | `color_space` |
| `ConvertRGB` | Ensure RGB | — |
| `Resize` | Smart resize | `width`, `height`, `smart_resize`, `min_pixels`, `max_pixels`, `patch_size`, `merge_size` |
| `Rescale` | Scale pixel values | `rescale_factor` |
| `Normalize` | Mean/std normalization | `mean`, `std` |
| `PatchImage` | Extract patches | `patch_size`, `temporal_patch_size`, `merge_size` |

---

## Generating from HuggingFace config

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_id)
ip = processor.image_processor

processor_config = {
    "processor": {
        "name": "qwen2_5_image_processor",
        "transforms": [
            {"operation": {"name": "decode_image", "type": "DecodeImage",
                           "attrs": {"color_space": "RGB"}}},
            {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
            {"operation": {"name": "resize", "type": "Resize",
                           "attrs": {
                               "width": 540, "height": 360, "smart_resize": 1,
                               "min_pixels": ip.size.get("shortest_edge", 3136),
                               "max_pixels": ip.size.get("longest_edge", 12845056),
                               "patch_size": ip.patch_size, "merge_size": ip.merge_size,
                           }}},
            {"operation": {"name": "rescale", "type": "Rescale",
                           "attrs": {"rescale_factor": ip.rescale_factor}}},
            {"operation": {"name": "normalize", "type": "Normalize", "attrs": {
                "mean": list(ip.image_mean), "std": list(ip.image_std),
                "qwen2_5_vl": 1,
            }}},
            {"operation": {"name": "patch_image", "type": "PatchImage", "attrs": {
                "patch_size": ip.patch_size,
                "temporal_patch_size": ip.temporal_patch_size,
                "merge_size": ip.merge_size,
            }}},
        ],
    }
}
```

---

## Writing image_processor.json from HuggingFace

```python
def _write_processor_config(processor, output_dir):
    ip = processor.image_processor
    config = {
        "processor": {
            "name": "qwen2_5_image_processor",
            "transforms": [
                {"operation": {"name": "decode_image", "type": "DecodeImage",
                               "attrs": {"color_space": "RGB"}}},
                {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
                {"operation": {"name": "resize", "type": "Resize", "attrs": {
                    "width": 540, "height": 360, "smart_resize": 1,
                    "min_pixels": ip.size.get("shortest_edge", 3136),
                    "max_pixels": ip.size.get("longest_edge", 12845056),
                    "patch_size": ip.patch_size, "merge_size": ip.merge_size,
                }}},
                {"operation": {"name": "rescale", "type": "Rescale",
                               "attrs": {"rescale_factor": ip.rescale_factor}}},
                {"operation": {"name": "normalize", "type": "Normalize", "attrs": {
                    "mean": list(ip.image_mean), "std": list(ip.image_std),
                    "qwen2_5_vl": 1,
                }}},
                {"operation": {"name": "patch_image", "type": "PatchImage", "attrs": {
                    "patch_size": ip.patch_size,
                    "temporal_patch_size": ip.temporal_patch_size,
                    "merge_size": ip.merge_size,
                }}},
            ],
        }
    }
    with open(os.path.join(output_dir, "image_processor.json"), "w") as f:
        json.dump(config, f, indent=2)
```
