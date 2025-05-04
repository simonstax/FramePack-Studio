import json
from pathlib import Path
from typing import Dict, Any, Optional
import os

class Settings:
    def __init__(self):
        self.settings_file = Path.home() / ".framepack" / "settings.json"
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        self.default_settings = {
            "output_dir": str(Path.home() / "Videos" / "FramePack"),
            "metadata_dir": str(Path.home() / "Videos" / "FramePack" / "metadata"),
            "lora_dir": str(Path.home() / "FramePack" / "loras"),
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

    def save_settings(self) -> bool:
        """Save current settings to file"""
        try:
            # Ensure directories exist
            os.makedirs(self.settings["output_dir"], exist_ok=True)
            os.makedirs(self.settings["metadata_dir"], exist_ok=True)
            os.makedirs(self.settings["lora_dir"], exist_ok=True)
            
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False

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
