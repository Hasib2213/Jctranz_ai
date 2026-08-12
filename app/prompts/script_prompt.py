from app.models.ai_models import ScriptGenerationRequest


def build_script_prompt(payload: ScriptGenerationRequest) -> str:
    target_audience = payload.target_audience or "general social media buyers"
    creator_prompt = payload.creator_prompt or "Create a clear, high-converting product promo."

    return f"""
Create a short promotional UGC video script for a product.

Product name:
{payload.product_name}

Product description:
{payload.product_description}

Creator direction:
{creator_prompt}

Target audience:
{target_audience}

Tone:
{payload.tone}

Video duration:
About {payload.duration_seconds} seconds.

Requirements:
- Write only the creator-facing spoken script.
- Make it natural, concise, and suitable for a short UGC/product reel.
- Include a strong hook in the first sentence.
- Mention the product benefit clearly.
- End with a simple call to action.
- Do not include scene labels, markdown, hashtags, or production notes.
""".strip()
