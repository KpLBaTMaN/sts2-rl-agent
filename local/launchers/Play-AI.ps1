$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location -LiteralPath $root
$python = Join-Path $root '.venv/Scripts/python.exe'
$model = Join-Path $root 'output/combat_ppo/best_model/best_model.zip'
$steam = 'C:/Program Files (x86)/Steam/steam.exe'
if (!(Test-Path -LiteralPath $model)) { throw 'No trained model found. Run Train.cmd first.' }
if (Get-Process SlayTheSpire2 -ErrorAction SilentlyContinue) { throw 'Close Slay the Spire 2 before starting the AI launcher.' }
Start-Process -FilePath $steam -ArgumentList '-applaunch','2868840','--sts2-rl' -WindowStyle Hidden | Out-Null
Write-Host 'If the game asks, enable mods. AI saves use modded/rl-agent profiles.'
Write-Host 'Close the game or press Ctrl+C here to stop the agent.'
& $python -m sts2_env.bridge.agent_runner --model-path $model --verbose --exit-on-disconnect --record-replay output/last-game-replay.json
