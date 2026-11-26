#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev"
#endif

[Setup]
AppID={{0E7879C6-1C84-42DD-9D3D-D457A4041157}}
AppName=Papylio
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\Papylio
DefaultGroupName=Papylio
OutputDir=..\..\dist
OutputBaseFilename=Papylio setup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Files]
Source: "..\..\dist\papylio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Papylio"; Filename: "{app}\papylio.exe"
Name: "{autodesktop}\Papylio"; Filename: "{app}\papylio.exe"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\papylio.exe"; Description: "Launch Papylio"; Flags: postinstall nowait skipifsilent