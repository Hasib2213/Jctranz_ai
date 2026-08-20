from app.models.ai_models import ScriptGenerationRequest


def build_script_prompt(payload: ScriptGenerationRequest) -> str:
    image_lines = ""
    if payload.product_images:
        image_lines = "\n".join(f"- {image}" for image in payload.product_images)

    return f"""
Create a short promotional UGC video script for a product.

Product name:
{payload.product_name}

Product description:
{payload.product_description}

Product reference images:
{image_lines or "None provided."}

Video duration:
About {payload.time_seconds} seconds.

Requirements:
- Write only the creator-facing spoken script.
- Make it natural, concise, and suitable for a short UGC/product reel.
- Include a strong hook in the first sentence.
- Mention the product benefit clearly.
- End with a simple call to action.
- Do not include scene labels, markdown, hashtags, or production notes.
""".strip()
