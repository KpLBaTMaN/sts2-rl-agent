extends SceneTree
## The bridge's code is in its separate C# DLL. Its resource pack needs only
## the manifest read by the game's mod loader: no project config or scripts.

func _initialize() -> void:
    var args := OS.get_cmdline_user_args()
    if args.size() != 2:
        printerr("Expected manifest source and output PCK paths")
        quit(2)
        return
    var manifest = JSON.parse_string(FileAccess.get_file_as_string(args[0]))
    if not manifest is Dictionary or manifest.get("pck_name") != "STS2BridgeMod":
        printerr("Invalid bridge manifest")
        quit(2)
        return
    var packer := PCKPacker.new()
    var result := packer.pck_start(args[1])
    if result == OK:
        result = packer.add_file("res://mod_manifest.json", args[0])
    if result == OK:
        result = packer.flush(true)
    if result != OK:
        printerr("Bridge manifest packaging failed: ", result)
        quit(1)
        return
    print("Bridge resource pack contains only mod_manifest.json")
    quit(0)
