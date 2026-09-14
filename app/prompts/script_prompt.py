from app.models.ai_models import ScriptGenerationRequest


def build_script_prompt(payload: ScriptGenerationRequest) -> str:
    image_lines = ""
    if payload.product_images:
        image_lines = "\n".join(f"- {image}" for image in payload.product_images)

    return f"""
Create a short promotional UGC video script for a product.
Return ONLY valid JSON with this exact shape:
{{
  "scenes": [
    {{
      "sequence": 1,
      "time": "0-3 sec",
      "visual": "Close-up of the product...",
      "voiceover": "Spoken line..."
    }}
  ]
}}
Do not add markdown, code fences, or any other text.

Product name:
{payload.product_name}

Product description:
{payload.product_description}

Product reference images:
{image_lines or "None provided."}

Video duration:
About {payload.time_seconds} seconds.

Requirements:
- Split the video into clear time blocks that cover the full duration.
- Create a `scenes` array in ascending `sequence` order.
- Make `time` simple and easy for the frontend to read.
- Make `visual` describe the scene, action, or b-roll in each block.
- Make `voiceover` contain only the spoken line for that block.
- Keep it natural, concise, and suitable for a short UGC/product reel.
- Include a strong hook near the beginning.
- Mention the product benefit clearly.
- End with a simple call to action.
""".strip()


def build_script_regeneration_prompt(
    product_name: str,
    product_description: str,
    product_images: list[str] | None,
    time_seconds: int,
    current_script: str,
) -> str:
    image_lines = ""
    if product_images:
        image_lines = "\n".join(f"- {image}" for image in product_images)

    return f"""
Rewrite the promotional UGC script below so it feels fresher, tighter, and more persuasive.
Keep the same product, core offer, and approximate duration.
Use the current script as the starting point, but improve the hook, pacing, clarity, and conversion.
Do not copy the current script sentence-for-sentence.
Return ONLY valid JSON with this exact shape:
{{
  "scenes": [
    {{
      "sequence": 1,
      "time": "0-3 sec",
      "visual": "Close-up of the product...",
      "voiceover": "Spoken line..."
    }}
  ]
}}
Do not add markdown, code fences, or any other text.

Product name:
{product_name}

Product description:
{product_description}

Product reference images:
{image_lines or "None provided."}

Video duration:
About {time_seconds} seconds.

Current script:
{current_script}

Requirements:
- Split the video into clear time blocks that cover the full duration.
- Keep the same scene order unless a better order clearly improves flow.
- Create a `scenes` array in ascending `sequence` order.
- Keep the same number of scenes unless a small change clearly improves pacing.
- Make `time` simple and easy for the frontend to read.
- Make `visual` describe the scene, action, or b-roll in each block.
- Make `voiceover` contain only the spoken line for that block.
- Keep it concise and suitable for a short UGC/product reel.
- Preserve factual product details and the call to action.
""".strip()
