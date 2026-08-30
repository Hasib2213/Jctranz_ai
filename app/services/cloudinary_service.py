import logging
import cloudinary
import cloudinary.uploader
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class CloudinaryService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.cloud_name = self.settings.cloudinary_cloud_name
        self.api_key = self.settings.cloudinary_api_key
        self.api_secret = self.settings.cloudinary_api_secret
        self.folder = self.settings.cloudinary_folder

        if self.is_configured:
            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
                secure=True,
            )

    @property
    def is_configured(self) -> bool:
        return bool(self.cloud_name and self.api_key and self.api_secret)

    def upload_media(
        self,
        file_source: str,
        resource_type: str = "auto",
        public_id: str | None = None,
        folder_override: str | None = None,
    ) -> dict:
        """
        Uploads a media URL, base64 string, or local file path to Cloudinary Object Storage.
        Returns a dictionary containing secure_url, public_id, asset_id, format, etc.
        If Cloudinary is not configured, returns the original source gracefully.
        """
        if not self.is_configured:
            logger.warning(
                "Cloudinary credentials are not configured. Returning original media URL/source."
            )
            return {
                "secure_url": file_source,
                "public_id": public_id or "",
                "cdn_provider": "fallback_direct",
                "is_uploaded": False,
            }

        target_folder = folder_override or self.folder

        upload_options = {
            "folder": target_folder,
            "resource_type": resource_type,
            "overwrite": True,
        }

        if public_id:
            upload_options["public_id"] = public_id

        try:
            response = cloudinary.uploader.upload(file_source, **upload_options)
            return {
                "secure_url": response.get("secure_url"),
                "public_id": response.get("public_id"),
                "asset_id": response.get("asset_id"),
                "resource_type": response.get("resource_type"),
                "format": response.get("format"),
                "bytes": response.get("bytes"),
                "cdn_provider": "cloudinary",
                "is_uploaded": True,
                "raw_response": response,
            }
        except Exception as exc:
            logger.error(f"Failed to upload media to Cloudinary: {exc}")
            raise RuntimeError(f"Cloudinary media upload failed: {exc}") from exc

    def upload_video(self, video_url: str, job_id: str | None = None) -> dict:
        """Helper specifically for uploading generated videos to Cloudinary CDN."""
        public_id = f"video_{job_id}" if job_id else None
        return self.upload_media(
            file_source=video_url,
            resource_type="video",
            public_id=public_id,
        )

    def upload_image(self, image_source: str, job_id: str | None = None) -> dict:
        """Helper for uploading images to Cloudinary CDN."""
        public_id = f"image_{job_id}" if job_id else None
        return self.upload_media(
            file_source=image_source,
            resource_type="image",
            public_id=public_id,
        )
