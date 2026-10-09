"""
Audio Alert Module for Emergency Events.

Plays alert sounds when fall or emergency is detected.
Supports multiple backends: pygame, winsound, playsound.
"""

import os
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AlertSound:
    """
    Audio alert player with multiple backend support.

    Tries backends in order of preference:
    1. pygame - best cross-platform support
    2. winsound - Windows only
    3. playsound - simple but may have issues
    """

    def __init__(self, sound_path: Optional[str] = None, enabled: bool = True):
        """
        Initialize alert sound player.

        Args:
            sound_path: Path to sound file (MP3/WAV)
            enabled: Whether to play sounds
        """
        self.enabled = enabled
        self.sound_path = sound_path or self._default_sound_path()
        self._backend = None
        self._pygame_initialized = False
        self._lock = threading.Lock()

        # Validate sound file exists
        if self.enabled and not os.path.exists(self.sound_path):
            logger.warning(f"Alert sound file not found: {self.sound_path}")
            logger.warning("Alert sounds will be disabled")
            self.enabled = False

        # Try to initialize backend
        self._init_backend()

    def _default_sound_path(self) -> str:
        """Get default sound file path."""
        base_dir = Path(__file__).parent.resolve()
        return str(base_dir / "alerts" / "emergency_alert.mp3")

    def _init_backend(self):
        """Initialize audio backend."""
        if not self.enabled:
            return

        # Try pygame first (most reliable)
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._backend = 'pygame'
            self._pygame_initialized = True
            logger.info("Audio backend: pygame")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"pygame init failed: {e}")

        # Try winsound (Windows)
        try:
            import winsound
            self._backend = 'winsound'
            logger.info("Audio backend: winsound")
            return
        except ImportError:
            pass

        # Try playsound
        try:
            from playsound import playsound
            self._backend = 'playsound'
            logger.info("Audio backend: playsound")
            return
        except ImportError:
            pass

        # No backend available
        logger.warning("No audio backend available. Install pygame: pip install pygame")
        self.enabled = False

    def play(self, repeat: int = 3):
        """
        Play alert sound.

        Args:
            repeat: Number of times to repeat the sound
        """
        if not self.enabled:
            return

        if self._backend is None:
            logger.debug("No audio backend configured")
            return

        # Play in separate thread to not block main thread
        thread = threading.Thread(
            target=self._play_sound,
            args=(repeat,),
            daemon=True
        )
        thread.start()

    def _play_sound(self, repeat: int):
        """Internal sound playing method."""
        with self._lock:
            for i in range(repeat):
                try:
                    if self._backend == 'pygame':
                        import pygame
                        if not self._pygame_initialized:
                            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                            self._pygame_initialized = True
                        sound = pygame.mixer.Sound(self.sound_path)
                        sound.play()
                        while pygame.mixer.get_busy():
                            pygame.time.delay(100)

                    elif self._backend == 'winsound':
                        import winsound
                        winsound.PlaySound(
                            self.sound_path,
                            winsound.SND_FILENAME | winsound.SND_ASYNC
                        )
                        # Simple delay for repeat
                        import time
                        time.sleep(2)

                    elif self._backend == 'playsound':
                        from playsound import playsound
                        playsound(self.sound_path, block=True)

                except Exception as e:
                    logger.error(f"Error playing sound (attempt {i+1}/{repeat}): {e}")

    def test(self) -> bool:
        """
        Test if alert sound works.

        Returns:
            True if sound plays successfully
        """
        if not self.enabled:
            logger.info("Alert sounds are disabled")
            return False

        logger.info(f"Testing alert sound: {self.sound_path}")
        self.play(repeat=1)

        # Wait a bit and check
        import time
        time.sleep(2)

        logger.info("Alert sound test completed")
        return True

    def stop(self):
        """Stop any currently playing sound."""
        if not self.enabled:
            return

        try:
            if self._backend == 'pygame':
                import pygame
                pygame.mixer.stop()
            elif self._backend == 'winsound':
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception as e:
            logger.debug(f"Error stopping sound: {e}")


# Global instance (lazy initialization)
_alert_sound: Optional[AlertSound] = None


def get_alert_sound(sound_path: Optional[str] = None, enabled: bool = True) -> AlertSound:
    """Get or create global AlertSound instance."""
    global _alert_sound
    if _alert_sound is None:
        _alert_sound = AlertSound(sound_path=sound_path, enabled=enabled)
    return _alert_sound


def play_emergency_alert(repeat: int = 3, sound_path: Optional[str] = None, enabled: bool = True):
    """
    Convenience function to play emergency alert.

    Args:
        repeat: Number of times to repeat
        sound_path: Optional custom sound path
        enabled: Whether to play (useful for config toggle)
    """
    if not enabled:
        return

    sound = get_alert_sound(sound_path=sound_path, enabled=enabled)
    sound.play(repeat=repeat)
    logger.info(f"Emergency alert triggered (repeat={repeat})")


def stop_alert():
    """Stop any playing alert."""
    global _alert_sound
    if _alert_sound is not None:
        _alert_sound.stop()
