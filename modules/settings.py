import json
from pathlib import Path
from typing import Dict, Any, Optional
import os

class Settings:
    def __init__(self):
        # Get the project root directory (where settings.py is located)
        project_root = Path(__file__).parent.parent
        
        self.settings_file = project_root / ".framepack" / "settings.json"
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Set default paths relative to project root
        self.default_settings = {
            "output_dir": str(project_root / "outputs"),
            "metadata_dir": str(project_root / "outputs"),
            "lora_dir": str(project_root / "loras"),
            "gradio_temp_dir": str(project_root / "temp"),
            "auto_save_settings": True,
            "gradio_theme": "default"
        }
        self.settings = self.load_settings()

    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file or return defaults"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Merge with defaults to ensure all settings exist
                    settings = self.default_settings.copy()
                    settings.update(loaded_settings)
                    return settings
            except Exception as e:
                print(f"Error loading settings: {e}")
                return self.default_settings.copy()
        return self.default_settings.copy()

    def save_settings(self, output_dir, metadata_dir, lora_dir, gradio_temp_dir, auto_save_settings, gradio_theme="default"):
        """Save settings to file"""
        self.settings = {
            "output_dir": output_dir,
            "metadata_dir": metadata_dir,
            "lora_dir": lora_dir,
            "gradio_temp_dir": gradio_temp_dir,
            "auto_save_settings": auto_save_settings,
            "gradio_theme": gradio_theme
        }
        
        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        os.makedirs(lora_dir, exist_ok=True)
        os.makedirs(gradio_temp_dir, exist_ok=True)
        
        # Save to file
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=4)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a setting value"""
        self.settings[key] = value
        if self.settings.get("auto_save_settings", True):
            self.save_settings()

    def update(self, settings: Dict[str, Any]) -> None:
        """Update multiple settings at once"""
        self.settings.update(settings)
        if self.settings.get("auto_save_settings", True):
            self.save_settings()
