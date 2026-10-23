import os
import webbrowser
from datetime import datetime
import platform

def find_app_shortcut(app_name):
    program_data = os.environ.get('PROGRAMDATA', r'C:\\ProgramData')
    app_data = os.environ.get('APPDATA')
    
    start_menu_dirs = [
        os.path.join(program_data, r'Microsoft\Windows\Start Menu\Programs')
    ]
    if app_data:
        start_menu_dirs.append(os.path.join(app_data, r'Microsoft\Windows\Start Menu\Programs'))
    
    app_name_lower = app_name.lower()
    
    for menu_dir in start_menu_dirs:
        if not os.path.exists(menu_dir): continue
        for root, dirs, files in os.walk(menu_dir):
            for file in files:
                if file.endswith('.lnk'):
                    name = file[:-4].lower()
                    if app_name_lower in name:
                        return os.path.join(root, file)
    return None

def open_app(entities):
    app_name = entities.get('app_name', None) if entities else None
    if not app_name:
        return "Please tell me which app to open. Example: 'open chrome' or 'launch spotify'."

    # Strip common trigger verbs from app_name in case they leaked in
    TRIGGER_VERBS = ["open", "launch", "start", "run", "fire up", "bring up", "boot up",
                     "please", "can you", "could you", "execute", "initiate"]
    app_name_lower = app_name.lower().strip()
    for verb in TRIGGER_VERBS:
        if app_name_lower.startswith(verb + " "):
            app_name_lower = app_name_lower[len(verb)+1:].strip()
    
    # System executable shortcuts map
    app_map = {
        "calculator": "calc.exe",       "calc": "calc.exe",
        "command prompt": "cmd.exe",    "cmd": "cmd.exe",
        "powershell": "powershell.exe",
        "notepad": "notepad.exe",
        "paint": "mspaint.exe",
        "task manager": "taskmgr.exe",
        "file explorer": "explorer.exe","explorer": "explorer.exe",
        "settings": "ms-settings:",
        "chrome": "chrome",
        "google chrome": "chrome",
        "firefox": "firefox",
        "edge": "msedge",
        "brave": "brave",
        "spotify": "spotify",
        "discord": "discord",
        "steam": "steam",
        "vlc": "vlc",
        "word": "winword",
        "excel": "excel",
        "powerpoint": "powerpnt",
        "vscode": "code",
        "vs code": "code",
        "visual studio code": "code",
        "whatsapp": "whatsapp",
        "telegram": "telegram",
    }

    executable = app_map.get(app_name_lower)
    if executable:
        try:
            os.system(f'start {executable}')
            return f"Opening {app_name_lower.title()} for you!"
        except Exception:
            pass

    # Scan Windows Start Menu shortcuts
    shortcut = find_app_shortcut(app_name_lower)
    if shortcut:
        try:
            os.startfile(shortcut)
            return f"Opened {app_name_lower.title()} via Start Menu shortcut."
        except Exception as e:
            return f"Found but failed to open {app_name_lower}: {e}"

    # Shell fallback  
    os.system(f'start "" "{app_name_lower}"')
    return f"Attempted to start '{app_name_lower}'. If it didn't open, check the app name."

def search_web(entities):
    query = entities.get('query', None) if entities else None
    if not query: return "What should I search for?"
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    webbrowser.open(url)
    return f"Searching the web for '{query}'..."

def get_time(entities=None):
    now = datetime.now()
    return f"The current system time is {now.strftime('%I:%M %p')}."
    
def get_date(entities=None):
    now = datetime.now()
    return f"Today's date is {now.strftime('%B %d, %Y')}."

def get_system_info(entities=None):
    return f"Your system runs {platform.system()} Version {platform.version()} / {platform.release()} on {platform.machine()} architecture."

def play_music(entities):
    song = entities.get('song_name', '').strip()
    song_lower = song.lower()
    
    # Check if they specifically want Spotify or just started with it
    if "spotify" in song_lower:
        clean_song = song_lower.replace("on spotify", "").replace("on music", "").replace("spotify", "").strip()
        if not clean_song or clean_song in ["music", "play"]:
             return open_app({"app_name": "spotify"})
        # Open Spotify search if it was a specific song - but since we don't have a spotify search function, we stick to launching spotify
        # Or we could use a web URL
        return open_app({"app_name": "spotify"})

    if not song or song_lower in ["music", "some music", "song"]:
        return open_app({"app_name": "spotify"})
    
    # If they want a specific song on YouTube
    if "youtube" in song_lower:
        clean_song = song_lower.replace("on youtube", "").replace("youtube", "").strip()
        if not clean_song:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube..."
        url = f"https://www.youtube.com/results?search_query={clean_song.replace(' ', '+')}"
        webbrowser.open(url)
        return f"Searching YouTube for '{clean_song.title()}'..."
    
    # Default to YouTube for queries like 'play Bohemian Rhapsody'
    url = f"https://www.youtube.com/results?search_query={song.replace(' ', '+')}"
    webbrowser.open(url)
    return f"Playing '{song.title()}' on YouTube!"

ACTION_MAP = {
    "open_app": open_app,
    "search_web": search_web,
    "get_time": get_time,
    "get_date": get_date,
    "get_system_info": get_system_info,
    "play_music": play_music
}

def execute_action(action_name, entities=None):
    """Executes mapped actions dynamically via dictionary."""
    if action_name in ACTION_MAP:
        return ACTION_MAP[action_name](entities)
    return None
