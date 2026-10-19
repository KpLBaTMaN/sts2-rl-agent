using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace STS2BridgeMod;

// Watch mode is opt-in and uses a separate save profile from normal play.
public static class WatchSession
{
    public static bool Enabled => CommandLineHelper.HasArg("sts2-watch");
    public static string Seed = "42";
    public static string Character = "Ironclad";
    public static int Ascension;
    public static bool StartWithNeow;
    public static bool ActOneOnly = true;
    public static volatile float Speed = 1;
    public static readonly CancellationTokenSource Cancellation = new();
    public static TimeSpan Timeout(TimeSpan original) => Enabled ? TimeSpan.FromHours(24) : original;

    public static void Load()
    {
        if (!Enabled) return;
        string path = Path.Combine(Path.GetDirectoryName(OS.GetExecutablePath())!, "mods", "STS2BridgeMod", "watch-session.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = doc.RootElement;
        Seed = root.GetProperty("seed").GetString()!;
        Character = root.GetProperty("character").GetString()!;
        Ascension = root.GetProperty("ascension").GetInt32();
        StartWithNeow = root.GetProperty("start_with_neow").GetBoolean();
        ActOneOnly = root.GetProperty("task").GetString() == "act1";
        if (string.IsNullOrWhiteSpace(Seed) || Ascension < 0 || Ascension > 9)
            throw new InvalidOperationException("Invalid watch session configuration");
        Logger.Log($"[Watch] Seed={Seed}, Character={Character}, Ascension={Ascension}, Opening={StartWithNeow}");
    }

    public static string Enrich(string json)
    {
        if (!Enabled) return json;
        JsonObject payload = JsonNode.Parse(json)!.AsObject();
        var run = RunManager.Instance.DebugOnlyGetState();
        var player = run == null ? null : LocalContext.GetMe(run);
        if (run == null || player == null) return json;
        payload["run_state"] = new JsonObject {
            ["act_index"] = run.CurrentActIndex,
            // The real game numbers the opening point as floor 1; the sim uses 0.
            ["act_floor"] = Math.Max(0, run.ActFloor - (run.CurrentActIndex == 0 ? 1 : 0)),
            ["floor"] = run.TotalFloor, ["hp"] = player.Creature.CurrentHp,
            ["max_hp"] = player.Creature.MaxHp, ["gold"] = player.Gold,
            ["deck_size"] = player.Deck.Cards.Count, ["relic_count"] = player.Relics.Count,
            ["potion_count"] = player.Potions.Count(), ["max_potion_slots"] = player.MaxPotionCount,
            ["ascension"] = run.AscensionLevel, ["seed"] = run.Rng.StringSeed,
            ["character"] = player.Character.Id.Entry,
            ["room"] = run.CurrentRoom?.RoomType.ToString() ?? "Map",
            ["in_combat"] = CombatManager.Instance.IsInProgress,
        };
        // Keep the last combat state available for reward/shop inputs, as in training.
        if (player.PlayerCombatState != null && run.CurrentRoom?.RoomType is RoomType.Monster or RoomType.Elite or RoomType.Boss)
            payload["combat_snapshot"] = JsonNode.Parse(new RlCombatHandler().SerializeCombatState(player));
        if (run.Map != null) {
            payload["map_nodes"] = JsonSerializer.SerializeToNode(run.Map.GetAllMapPoints()
                .Select(p => new { row = p.coord.row, col = p.coord.col, type = p.PointType.ToString() }));
        }
        return payload.ToJsonString();
    }
}

[HarmonyPatch(typeof(RunState), nameof(RunState.CreateForNewRun))]
public static class WatchRunConfigurationPatch
{
    [HarmonyPrefix]
    static void Prefix(ref string seed, ref int ascensionLevel)
    {
        if (!WatchSession.Enabled) return;
        seed = WatchSession.Seed;
        ascensionLevel = WatchSession.Ascension;
    }
}

// Bypass the opening event without granting a blessing when the simulator starts at the map.
[HarmonyPatch(typeof(RunManager), nameof(RunManager.EnterMapCoord), new Type[] { typeof(MapCoord) })]
public static class WatchOpeningPatch
{
    [HarmonyPrefix]
    static bool Prefix(RunManager __instance, MapCoord coord, ref Task __result)
    {
        if (!WatchSession.Enabled || WatchSession.StartWithNeow) return true;
        var run = __instance.DebugOnlyGetState();
        if (run.CurrentActIndex != 0 || run.TotalFloor > 0 ||
            !coord.Equals(run.Map.StartingMapPoint.coord)) return true;
        Logger.Log("[Watch] Skipping opening event to match start_with_neow=false");
        // Treat the opening point as passed, so the next vote targets its children.
        run.AddVisitedMapCoord(coord);
        __result = __instance.EnterRoom(new MapRoom());
        return false;
    }
}
