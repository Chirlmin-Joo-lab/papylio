# dmg_settings.py
app = '../../dist/mac/papylio.app'
appname = 'Papylio'

format = 'UDZO'          # compressed DMG (this is the DMG format, unrelated to your app-startup compression issue)
size = None              # auto-calculated
files = [app]
symlinks = {'Applications': '/Applications'}

# Icon size and layout
icon_size = 128
icon_locations = {
    'papylio.app': (150, 190),
    'Applications': (450, 190),
}

window_rect = ((400, 300), (600+128/4, 400+128/4))  # position, size
default_view = 'icon-view'
show_icon_preview = True

# Optional custom background — omit background line for plain white
# background = 'path/to/background.png'