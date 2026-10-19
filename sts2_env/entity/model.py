"""SpireFormer: an entity Transformer policy with pointer-style action heads.

Pipeline::

    IDs + details ──► embeddings ──► Transformer (items attend to each other)
                                         │
                 ┌───────────────────────┼──────────────────────────┐
           global token            card / potion / choice      enemy tokens
                 │                  tokens (what to use)       (who to target)
                 ▼                         ▼                          ▼
        end turn + value        score(card)  score(card, enemy)  score(potion, enemy)

Each action logit is produced by scoring the item it refers to, so "play card
in slot 3" is computed from that card's own token instead of a fixed output
neuron.  Outside fights, every on-screen choice (map node, offered card, shop
item, rest/event option...) is an option token that carries its action index;
its score is written there.  Scores follow the environments' existing action
layouts (115 combat actions, 157 run actions) so the usual legality masks apply
unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from sts2_env.core.constants import ACTION_SPACE_SIZE, MAX_ENEMIES, MAX_HAND_SIZE, MAX_POTION_SLOTS
from sts2_env.entity.observation import (
    CARD_FEATURES,
    ENEMY_FEATURES,
    GLOBAL_FEATURES,
    MAX_CHOICES,
    OPTION_FEATURES,
    POTION_FEATURES,
    RUN_FEATURES,
    VOCAB_SIZES,
)

# Token-type IDs, so the Transformer knows what kind of item each token is.
GLOBAL, HAND, PILE, ENEMY, POTION, RELIC, CHOICE, OPTION = range(8)


@dataclass(frozen=True)
class SpireFormerConfig:
    d_model: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.0
    n_actions: int = ACTION_SPACE_SIZE


def _mlp(inputs: int, hidden: int, outputs: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class SpireFormer(nn.Module):
    def __init__(self, config: SpireFormerConfig = SpireFormerConfig()):
        super().__init__()
        self.config = config
        d = config.d_model
        self.card_embedding = nn.Embedding(VOCAB_SIZES["card"], d, padding_idx=0)
        self.power_embedding = nn.Embedding(VOCAB_SIZES["power"], d, padding_idx=0)
        self.monster_embedding = nn.Embedding(VOCAB_SIZES["monster"], d, padding_idx=0)
        self.potion_embedding = nn.Embedding(VOCAB_SIZES["potion"], d, padding_idx=0)
        self.relic_embedding = nn.Embedding(VOCAB_SIZES["relic"], d, padding_idx=0)
        self.pile_embedding = nn.Embedding(VOCAB_SIZES["pile"], d, padding_idx=0)
        self.name_embedding = nn.Embedding(VOCAB_SIZES["name"], d, padding_idx=0)
        self.option_kind_embedding = nn.Embedding(VOCAB_SIZES["option_kind"], d, padding_idx=0)
        self.token_type_embedding = nn.Embedding(8, d)

        self.global_details = nn.Linear(GLOBAL_FEATURES, d)
        self.run_details = nn.Linear(RUN_FEATURES, d)
        self.option_details = nn.Linear(OPTION_FEATURES, d)
        self.card_details = nn.Linear(CARD_FEATURES, d)
        self.enemy_details = nn.Linear(ENEMY_FEATURES, d)
        self.potion_details = nn.Linear(POTION_FEATURES, d)
        self.power_amount = nn.Linear(1, d)
        self.power_norm = nn.LayerNorm(d)

        layer = nn.TransformerEncoderLayer(
            d, config.heads, dim_feedforward=4 * d, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d)

        self.end_turn_head = _mlp(d, d)
        self.card_head = _mlp(2 * d, d)
        self.card_target_head = _mlp(3 * d, d)
        self.potion_head = _mlp(2 * d, d)
        self.potion_target_head = _mlp(3 * d, d)
        self.choice_head = _mlp(2 * d, d)
        self.option_head = _mlp(2 * d, d)
        self.value_head = _mlp(2 * d, d)

    def _pooled_powers(self, ids: torch.Tensor, amounts: torch.Tensor) -> torch.Tensor:
        """Sum each creature's buffs/debuffs into one vector (ID + stack size)."""
        tokens = self.power_embedding(ids) + self.power_amount(amounts.unsqueeze(-1))
        tokens = tokens * (ids > 0).unsqueeze(-1)
        return self.power_norm(tokens.sum(dim=-2))

    def _typed(self, tokens: torch.Tensor, kind: int) -> torch.Tensor:
        return tokens + self.token_type_embedding.weight[kind]

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits[B, n_actions], value[B])``; logits are not yet masked."""
        batch = obs["global"].shape[0]
        device = obs["global"].device

        global_token = (self.global_details(obs["global"]) + self.run_details(obs["run"])
                        + self._pooled_powers(obs["player_power_ids"], obs["player_power_amounts"]))
        hand = self.card_embedding(obs["hand_ids"]) + self.card_details(obs["hand_features"])
        pile = (self.card_embedding(obs["pile_ids"]) + self.card_details(obs["pile_features"])
                + self.pile_embedding(obs["pile_kinds"]))
        enemies = (self.monster_embedding(obs["enemy_ids"]) + self.enemy_details(obs["enemy_features"])
                   + self._pooled_powers(obs["enemy_power_ids"], obs["enemy_power_amounts"]))
        potions = self.potion_embedding(obs["potion_ids"]) + self.potion_details(obs["potion_features"])
        relics = self.relic_embedding(obs["relic_ids"])
        choices = (self.card_embedding(obs["choice_ids"]) + self.card_details(obs["choice_features"])
                   + self.pile_embedding(obs["choice_kinds"]))
        options = (self.option_kind_embedding(obs["option_kinds"]) + self.card_embedding(obs["option_cards"])
                   + self.relic_embedding(obs["option_relics"]) + self.potion_embedding(obs["option_potions"])
                   + self.name_embedding(obs["option_names"]) + self.option_details(obs["option_features"]))

        groups = [
            (self._typed(global_token.unsqueeze(1), GLOBAL), torch.ones(batch, 1, dtype=torch.bool, device=device)),
            (self._typed(hand, HAND), obs["hand_mask"]),
            (self._typed(pile, PILE), obs["pile_mask"]),
            (self._typed(enemies, ENEMY), obs["enemy_mask"]),
            (self._typed(potions, POTION), obs["potion_mask"]),
            (self._typed(relics, RELIC), obs["relic_mask"]),
            (self._typed(choices, CHOICE), obs["choice_mask"]),
            (self._typed(options, OPTION), obs["option_mask"]),
        ]
        tokens = torch.cat([g[0] for g in groups], dim=1)
        present = torch.cat([g[1] for g in groups], dim=1)
        encoded = self.final_norm(self.encoder(tokens, src_key_padding_mask=~present))

        sizes = [g[0].shape[1] for g in groups]
        (g_out, hand_out, _, enemy_out, potion_out, _, choice_out, option_out) = torch.split(encoded, sizes, dim=1)
        g_out = g_out.squeeze(1)

        pooled = (encoded * present.unsqueeze(-1)).sum(1) / present.sum(1, keepdim=True)
        value = self.value_head(torch.cat([g_out, pooled], dim=-1)).squeeze(-1)

        g_hand = g_out.unsqueeze(1).expand(-1, MAX_HAND_SIZE, -1)
        g_pair = g_out[:, None, None, :].expand(-1, MAX_HAND_SIZE, MAX_ENEMIES, -1)
        end_turn = self.end_turn_head(g_out)
        cards = self.card_head(torch.cat([hand_out, g_hand], -1)).squeeze(-1)
        card_pairs = torch.cat([
            hand_out.unsqueeze(2).expand(-1, -1, MAX_ENEMIES, -1),
            enemy_out.unsqueeze(1).expand(-1, MAX_HAND_SIZE, -1, -1),
            g_pair,
        ], -1)
        card_targets = self.card_target_head(card_pairs).squeeze(-1).reshape(batch, -1)

        g_pot = g_out.unsqueeze(1).expand(-1, MAX_POTION_SLOTS, -1)
        potion_self = self.potion_head(torch.cat([potion_out, g_pot], -1))
        potion_pairs = torch.cat([
            potion_out.unsqueeze(2).expand(-1, -1, MAX_ENEMIES, -1),
            enemy_out.unsqueeze(1).expand(-1, MAX_POTION_SLOTS, -1, -1),
            g_out[:, None, None, :].expand(-1, MAX_POTION_SLOTS, MAX_ENEMIES, -1),
        ], -1)
        potion_targets = self.potion_target_head(potion_pairs).squeeze(-1)
        # Per slot: [use untargeted, use on enemy 0..4], matching the action layout.
        potions_flat = torch.cat([potion_self, potion_targets], -1).reshape(batch, -1)

        normal = torch.cat([end_turn, cards, card_targets, potions_flat], dim=-1)

        g_choice = g_out.unsqueeze(1).expand(-1, MAX_CHOICES, -1)
        choice_scores = self.choice_head(torch.cat([choice_out, g_choice], -1)).squeeze(-1)
        unused = torch.zeros(batch, ACTION_SPACE_SIZE - 1 - MAX_CHOICES, device=device)
        choosing = torch.cat([end_turn, choice_scores, unused], dim=-1)

        combat_logits = torch.where(obs["pending_choice"], choosing, normal)

        # Run actions beyond the combat block start at 0 and are filled by option scores.
        n_actions = self.config.n_actions
        logits = torch.cat([combat_logits, torch.zeros(batch, n_actions - ACTION_SPACE_SIZE + 1, device=device)], -1)
        g_opt = g_out.unsqueeze(1).expand(-1, option_out.shape[1], -1)
        option_scores = self.option_head(torch.cat([option_out, g_opt], -1)).squeeze(-1)
        # Padded options write into a spare last column, which is then dropped.
        slots = torch.where(obs["option_mask"], obs["option_slots"], torch.full_like(obs["option_slots"], n_actions))
        logits = logits.scatter(1, slots, option_scores)[:, :n_actions]
        return logits, value

    def describe(self) -> dict:
        return dict(asdict(self.config), parameters=sum(p.numel() for p in self.parameters()))


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


assert 1 + MAX_HAND_SIZE * (1 + MAX_ENEMIES) + MAX_POTION_SLOTS * (1 + MAX_ENEMIES) == ACTION_SPACE_SIZE
assert 1 + MAX_CHOICES <= ACTION_SPACE_SIZE
