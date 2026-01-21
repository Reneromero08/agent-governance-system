"""
NOTIFIER MODULE
===============

Multiple notification methods for trading alerts:
1. Desktop toast (Windows)
2. Sound alert (beep)
3. Telegram bot (optional, needs config)
4. Discord webhook (optional, needs config)

All methods are best-effort - failures don't crash the bot.
"""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_FILE = Path(__file__).parent / "notifier_config.json"

@dataclass
class NotifierConfig:
    """Notification settings."""
    # Desktop
    desktop_enabled: bool = True
    sound_enabled: bool = True

    # Telegram (optional)
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Discord (optional)
    discord_enabled: bool = False
    discord_webhook_url: str = ""

    @classmethod
    def load(cls) -> "NotifierConfig":
        """Load config from file or create default."""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()

    def save(self):
        """Save config to file."""
        with open(CONFIG_FILE, 'w') as f:
            json.dump({
                "desktop_enabled": self.desktop_enabled,
                "sound_enabled": self.sound_enabled,
                "telegram_enabled": self.telegram_enabled,
                "telegram_bot_token": self.telegram_bot_token,
                "telegram_chat_id": self.telegram_chat_id,
                "discord_enabled": self.discord_enabled,
                "discord_webhook_url": self.discord_webhook_url,
            }, f, indent=2)


# =============================================================================
# NOTIFICATION METHODS
# =============================================================================

def notify_desktop(title: str, message: str, urgency: int = 3):
    """
    Windows toast notification.

    Uses win10toast if available, falls back to ctypes MessageBox.
    """
    try:
        # Try win10toast first (pip install win10toast)
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(
            title,
            message[:256],  # Limit length
            duration=10 if urgency >= 4 else 5,
            threaded=True
        )
        return True
    except ImportError:
        pass

    try:
        # Fallback: ctypes MessageBox (blocking but always works)
        import ctypes
        MB_ICONWARNING = 0x30
        MB_ICONERROR = 0x10
        icon = MB_ICONERROR if urgency >= 4 else MB_ICONWARNING
        ctypes.windll.user32.MessageBoxW(0, message, title, icon)
        return True
    except Exception:
        pass

    return False


def notify_sound(urgency: int = 3):
    """
    Play alert sound.

    Uses winsound on Windows.
    """
    try:
        import winsound

        if urgency >= 4:
            # Urgent: multiple beeps
            for _ in range(3):
                winsound.Beep(1000, 500)  # 1000Hz for 500ms
                winsound.Beep(800, 500)
        else:
            # Normal: single beep
            winsound.Beep(800, 1000)

        return True
    except Exception:
        # Fallback: print bell character
        print("\a" * (3 if urgency >= 4 else 1))
        return True


def notify_telegram(
    message: str,
    bot_token: str,
    chat_id: str
) -> bool:
    """
    Send Telegram message via bot.

    Setup:
    1. Message @BotFather on Telegram
    2. Send /newbot and follow prompts
    3. Copy the token
    4. Message your bot, then get chat_id from:
       https://api.telegram.org/bot<TOKEN>/getUpdates
    """
    if not bot_token or not chat_id:
        return False

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = json.dumps({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }).encode('utf-8')

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200

    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")
        return False


def notify_discord(
    message: str,
    webhook_url: str
) -> bool:
    """
    Send Discord message via webhook.

    Setup:
    1. Server Settings -> Integrations -> Webhooks
    2. New Webhook -> Copy Webhook URL
    """
    if not webhook_url:
        return False

    try:
        data = json.dumps({
            "content": message
        }).encode('utf-8')

        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in [200, 204]

    except Exception as e:
        print(f"[DISCORD ERROR] {e}")
        return False


# =============================================================================
# UNIFIED NOTIFIER
# =============================================================================

class Notifier:
    """
    Unified notification system.

    Sends alerts via all enabled channels.
    """

    def __init__(self, config: Optional[NotifierConfig] = None):
        self.config = config or NotifierConfig.load()

    def notify(
        self,
        title: str,
        message: str,
        urgency: int = 3,
        include_details: bool = True
    ):
        """
        Send notification via all enabled channels.

        Args:
            title: Alert title (short)
            message: Alert message (can be longer)
            urgency: 1-5 (5 = most urgent)
            include_details: Whether to include full message in all channels
        """
        results = {}

        # Desktop notification
        if self.config.desktop_enabled:
            results["desktop"] = notify_desktop(title, message, urgency)

        # Sound alert
        if self.config.sound_enabled:
            results["sound"] = notify_sound(urgency)

        # Telegram
        if self.config.telegram_enabled:
            telegram_msg = f"<b>{title}</b>\n\n{message}" if include_details else title
            results["telegram"] = notify_telegram(
                telegram_msg,
                self.config.telegram_bot_token,
                self.config.telegram_chat_id
            )

        # Discord
        if self.config.discord_enabled:
            discord_msg = f"**{title}**\n{message}" if include_details else title
            results["discord"] = notify_discord(
                discord_msg,
                self.config.discord_webhook_url
            )

        return results

    def test(self):
        """Send test notification to all enabled channels."""
        return self.notify(
            title="TEST ALERT",
            message="This is a test notification from Psychohistory Options Bot.",
            urgency=3
        )


# =============================================================================
# SETUP HELPER
# =============================================================================

def setup_notifications():
    """Interactive setup for notification preferences."""
    print("=" * 60)
    print("NOTIFICATION SETUP")
    print("=" * 60)

    config = NotifierConfig.load()

    # Desktop
    print("\n1. DESKTOP NOTIFICATIONS (Windows toast)")
    resp = input("   Enable desktop alerts? [Y/n]: ").strip().lower()
    config.desktop_enabled = resp != 'n'

    # Sound
    print("\n2. SOUND ALERTS")
    resp = input("   Enable sound alerts? [Y/n]: ").strip().lower()
    config.sound_enabled = resp != 'n'

    # Telegram
    print("\n3. TELEGRAM BOT (optional - sends to your phone)")
    print("   To set up:")
    print("   - Message @BotFather on Telegram")
    print("   - Send /newbot and follow prompts")
    print("   - Message your new bot")
    print("   - Visit: https://api.telegram.org/bot<TOKEN>/getUpdates")
    resp = input("   Enable Telegram? [y/N]: ").strip().lower()
    config.telegram_enabled = resp == 'y'

    if config.telegram_enabled:
        config.telegram_bot_token = input("   Bot token: ").strip()
        config.telegram_chat_id = input("   Chat ID: ").strip()

    # Discord
    print("\n4. DISCORD WEBHOOK (optional)")
    print("   To set up:")
    print("   - Server Settings -> Integrations -> Webhooks")
    print("   - New Webhook -> Copy URL")
    resp = input("   Enable Discord? [y/N]: ").strip().lower()
    config.discord_enabled = resp == 'y'

    if config.discord_enabled:
        config.discord_webhook_url = input("   Webhook URL: ").strip()

    # Save
    config.save()
    print(f"\nConfig saved to: {CONFIG_FILE}")

    # Test
    resp = input("\nSend test notification? [Y/n]: ").strip().lower()
    if resp != 'n':
        notifier = Notifier(config)
        results = notifier.test()
        print(f"Results: {results}")

    print("\nSetup complete!")
    return config


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_notifications()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        notifier = Notifier()
        results = notifier.test()
        print(f"Results: {results}")
    else:
        print("Usage:")
        print("  python notifier.py --setup    # Configure notifications")
        print("  python notifier.py --test     # Test notifications")
