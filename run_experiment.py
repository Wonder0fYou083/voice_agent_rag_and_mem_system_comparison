#!/usr/bin/env python3
"""Product-shaped memory/RAG benchmark built from a real voice-agent log.

This is a controlled local proxy experiment. It does NOT call the hosted
Mem0 or mem9 products. The two memory policies intentionally model two
different architectural choices:

* mem0_style: selective fact extraction + canonicalization + consolidation.
* mem9_style: event-preserving memory + lightweight de-duplication.

The retrieval layer is independently varied:

* current_rag: scoped character n-gram TF-IDF retrieval.
* graph_rag: semantic retrieval plus entity-link and neighbor boosting.
* sag: deterministic event/entity activation plus semantic and temporal fusion.

The interfaces and output schemas are designed so real product adapters can
replace the local policies later without changing the benchmark or metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = ROOT.parent / "upload" / "test_agent.service_78.log"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T[\d:]+\+\d{4})"
    r".*?agent\[\d+\]:\s+\d{4}/\d{2}/\d{2}\s+[\d:]+\s+"
    r"\[(?P<tag>[^\]]+)\]\s*(?P<message>.*)$"
)
QUOTED_TRANSCRIPT_RE = re.compile(r'(?:转写|transcript)="(?P<text>.*?)"')
LOOPBACK_RE = re.compile(r'(?:partial|final).*?text="(?P<text>.*?)"')


ALIASES = {
    "豆小饼": "豆小饼",
    "豆小包": "豆小饼",
    "窦小丙": "豆小饼",
    "做小饼": "豆小饼",
    "朱一闻": "朱一闻",
    "朱玉文": "朱一闻",
    "周宇文": "朱一闻",
    "毕文": "毕文",
    "Bill": "Bill",
    "比尔": "Bill",
    "Alice": "Alice",
    "爱丽丝": "Alice",
    "Charles": "Charles",
    "查尔斯": "Charles",
    "Kaiwen": "Kaiwen",
    "凯文": "Kaiwen",
    "亚索": "亚索",
    "卡牌大师": "卡牌大师",
    "卡牌": "卡牌大师",
    "锤石": "锤石",
    "大乱斗": "ARAM",
    "ARAM": "ARAM",
    "英雄联盟": "英雄联盟",
    "LOL": "英雄联盟",
    "COD": "COD",
    "使命召唤": "COD",
    "Valorant": "Valorant",
    "无畏契约": "Valorant",
}

GAME_IDS = {
    "英雄联盟": 1,
    "ARAM": 1,
    "COD": 2,
    "Valorant": 3,
}

TEAMMATE_IDS = {
    "Bill": 250,
    "Alice": 251,
    "Charles": 252,
    "朱一闻": 253,
    "毕文": 254,
}


@dataclass
class Event:
    event_id: str
    timestamp: str
    source: str
    speaker_type: str
    text: str
    event_type: str = "utterance"
    should_store: bool = False
    memory_kind: str = "memory"
    slot: str | None = None
    value: str | None = None
    canonical: str | None = None
    scope: str = "session"
    user_id: int = 105
    game_id: int | None = 1
    teammate_id: int | None = None
    entities: list[str] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)


@dataclass
class Memory:
    memory_id: str
    source_event_id: str
    timestamp: str
    content: str
    memory_kind: str
    slot: str | None
    value: str | None
    scope: str
    user_id: int
    game_id: int | None
    teammate_id: int | None
    entities: list[str]
    relations: list[dict]
    memory_system: str


@dataclass
class Query:
    query_id: str
    text: str
    category: str
    user_id: int = 105
    game_id: int | None = None
    teammate_id: int | None = None
    gold: list[tuple[str, str]] = field(default_factory=list)
    should_abstain: bool = False


def iso_ts(hhmmss: str) -> str:
    return f"2026-07-22T{hhmmss}+08:00"


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？、,.!?：:“”\"'（）()《》]", "", text)
    return text.lower()


def canonical_entities(text: str) -> list[str]:
    found: list[str] = []
    for alias, canonical in sorted(ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if alias.lower() in text.lower() and canonical not in found:
            found.append(canonical)
    return found


def parse_log(path: Path) -> tuple[list[Event], dict]:
    events: list[Event] = []
    raw_counts: defaultdict[str, int] = defaultdict(int)
    pending_loopback: Event | None = None
    event_number = 0

    def add_event(timestamp: str, source: str, speaker: str, text: str, kind: str) -> None:
        nonlocal event_number
        clean = text.strip()
        if not clean:
            return
        event_number += 1
        events.append(
            Event(
                event_id=f"real_{event_number:04d}",
                timestamp=timestamp,
                source=source,
                speaker_type=speaker,
                text=clean,
                event_type=kind,
                should_store=False,
                entities=canonical_entities(clean),
            )
        )

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            match = LINE_RE.match(raw_line)
            if not match:
                continue
            timestamp = datetime.strptime(
                match.group("timestamp"), "%Y-%m-%dT%H:%M:%S%z"
            ).isoformat()
            tag = match.group("tag")
            message = match.group("message")
            raw_counts[tag] += 1

            if tag == "MicCasual":
                transcript = QUOTED_TRANSCRIPT_RE.search(message)
                if transcript:
                    add_event(
                        timestamp,
                        "mic_normalized",
                        "user",
                        transcript.group("text"),
                        "utterance",
                    )

            elif tag == "Loopback":
                transcript = LOOPBACK_RE.search(message)
                if not transcript:
                    continue
                text = transcript.group("text").strip()
                candidate = Event(
                    event_id="",
                    timestamp=timestamp,
                    source="loopback",
                    speaker_type="teammate",
                    text=text,
                    event_type="utterance",
                    entities=canonical_entities(text),
                )
                if pending_loopback is None:
                    pending_loopback = candidate
                else:
                    prev_time = datetime.fromisoformat(pending_loopback.timestamp)
                    new_time = datetime.fromisoformat(timestamp)
                    close = (new_time - prev_time).total_seconds() <= 3.5
                    similar = (
                        SequenceMatcher(
                            None,
                            normalize_text(pending_loopback.text),
                            normalize_text(text),
                        ).ratio()
                        >= 0.35
                    )
                    if close and similar:
                        if len(text) >= len(pending_loopback.text):
                            pending_loopback = candidate
                    else:
                        add_event(
                            pending_loopback.timestamp,
                            pending_loopback.source,
                            pending_loopback.speaker_type,
                            pending_loopback.text,
                            pending_loopback.event_type,
                        )
                        pending_loopback = candidate

            elif tag == "LLM-req":
                command_match = re.search(
                    r"用户操作命令[^：]*：(?P<command>.*?)(?:\\n|$)", message
                )
                if command_match:
                    command = command_match.group("command").strip()
                    if command not in {"（无）", "(无)", "无"}:
                        add_event(timestamp, "llm_request", "user", command, "command")

    if pending_loopback is not None:
        add_event(
            pending_loopback.timestamp,
            pending_loopback.source,
            pending_loopback.speaker_type,
            pending_loopback.text,
            pending_loopback.event_type,
        )

    # The same durable preference appears in the real log in multiple ASR forms.
    # We label the canonical occurrence as a gold memory and leave the other
    # noisy forms in the background to test entity/duplicate behavior.
    durable_added = False
    for event in events:
        if "不许叫我主人" in event.text or "别叫我主人" in event.text:
            event.should_store = True
            event.slot = "teammate_253.address_preference"
            event.value = "不要称呼主人"
            event.canonical = "朱一闻不希望豆小饼称呼他为主人。"
            event.scope = "teammate_global"
            event.teammate_id = 253
            event.entities = ["朱一闻", "豆小饼"]
            event.relations = [
                {"subject": "朱一闻", "predicate": "dislikes_address", "object": "主人"}
            ]
            durable_added = True
            break

    stats = {
        "total_lines": sum(raw_counts.values()),
        "tag_counts": dict(sorted(raw_counts.items())),
        "semantic_events": len(events),
        "user_events": sum(e.speaker_type == "user" for e in events),
        "teammate_events": sum(e.speaker_type == "teammate" for e in events),
        "durable_real_events": int(durable_added),
        "first_semantic_event": min((e.timestamp for e in events), default=None),
        "last_semantic_event": max((e.timestamp for e in events), default=None),
    }
    if events:
        stats["semantic_duration_minutes"] = round(
            (
                datetime.fromisoformat(stats["last_semantic_event"])
                - datetime.fromisoformat(stats["first_semantic_event"])
            ).total_seconds()
            / 60,
            2,
        )
    return events, stats


def controlled_events() -> list[Event]:
    """Controlled facts injected into the real-noise timeline."""

    rows = [
        # id, time, text, slot, value, canonical, scope, game, teammate, entities, relations
        (
            "ctrl_001", "00:05:30", "以后默认用中文跟我交流。", "user.language",
            "中文", "Kaiwen偏好使用中文交流。", "global", None, None,
            ["Kaiwen"], [{"subject": "Kaiwen", "predicate": "prefers_language", "object": "中文"}],
        ),
        (
            "ctrl_002", "00:06:30", "回答尽量简短，我在打游戏没时间听长篇解释。", "user.reply_style",
            "简短", "Kaiwen在游戏中偏好简短回答。", "global", None, None,
            ["Kaiwen"], [{"subject": "Kaiwen", "predicate": "prefers_reply_style", "object": "简短"}],
        ),
        (
            "ctrl_003", "00:07:20", "我习惯叫这个AI豆小饼。", "agent.nickname",
            "豆小饼", "Kaiwen将AI称为豆小饼。", "global", None, None,
            ["Kaiwen", "豆小饼"], [{"subject": "Kaiwen", "predicate": "calls_agent", "object": "豆小饼"}],
        ),
        (
            "ctrl_004", "00:08:10", "大乱斗里我以前喜欢玩输出位。", "game_1.preferred_role",
            "输出", "Kaiwen以前在英雄联盟大乱斗中偏好输出位。", "user_game", 1, None,
            ["Kaiwen", "ARAM"], [{"subject": "Kaiwen", "predicate": "preferred_role", "object": "输出"}],
        ),
        (
            "ctrl_005", "00:18:50", "现在大乱斗我更喜欢玩坦克，别再推荐输出位了。", "game_1.preferred_role",
            "坦克", "Kaiwen当前在英雄联盟大乱斗中偏好坦克角色。", "user_game", 1, None,
            ["Kaiwen", "ARAM"], [{"subject": "Kaiwen", "predicate": "prefers_role", "object": "坦克"}],
        ),
        (
            "ctrl_006", "00:09:05", "我以前玩卡牌大师习惯出AP。", "game_1.twisted_fate_build",
            "AP", "Kaiwen以前在大乱斗玩卡牌大师时偏好AP出装。", "user_game", 1, None,
            ["Kaiwen", "ARAM", "卡牌大师"], [{"subject": "Kaiwen", "predicate": "preferred_build", "object": "AP"}],
        ),
        (
            "ctrl_007", "00:22:10", "我现在玩卡牌大师改成攻速AD出装。", "game_1.twisted_fate_build",
            "攻速AD", "Kaiwen当前在大乱斗玩卡牌大师时偏好攻速AD出装。", "user_game", 1, None,
            ["Kaiwen", "ARAM", "卡牌大师"], [{"subject": "Kaiwen", "predicate": "prefers_build", "object": "攻速AD"}],
        ),
        (
            "ctrl_008", "00:10:40", "和Bill打使命召唤时我习惯用狙击手。", "game_2.teammate_250.role",
            "狙击手", "Kaiwen与Bill玩COD时偏好使用狙击手。", "teammate_game", 2, 250,
            ["Kaiwen", "Bill", "COD"], [{"subject": "Kaiwen", "predicate": "prefers_role", "object": "狙击手"}],
        ),
        (
            "ctrl_009", "00:11:20", "我用狙的时候Bill通常帮我守近点。", "game_2.teammate_250.coordination",
            "Bill守近点", "Kaiwen在COD使用狙击手时，Bill负责保护近距离区域。", "teammate_game", 2, 250,
            ["Kaiwen", "Bill", "COD"], [{"subject": "Bill", "predicate": "covers_close_range_for", "object": "Kaiwen"}],
        ),
        (
            "ctrl_010", "00:12:20", "和Bill玩无畏契约时我喜欢打突破，不玩狙。", "game_3.teammate_250.role",
            "突破", "Kaiwen与Bill玩Valorant时偏好突破位。", "teammate_game", 3, 250,
            ["Kaiwen", "Bill", "Valorant"], [{"subject": "Kaiwen", "predicate": "prefers_role", "object": "突破"}],
        ),
        (
            "ctrl_011", "00:13:00", "和Alice玩无畏契约时我通常打控场。", "game_3.teammate_251.role",
            "控场", "Kaiwen与Alice玩Valorant时偏好控场位。", "teammate_game", 3, 251,
            ["Kaiwen", "Alice", "Valorant"], [{"subject": "Kaiwen", "predicate": "prefers_role", "object": "控场"}],
        ),
        (
            "ctrl_012", "00:13:40", "和Charles玩大乱斗时我喜欢让他玩前排。", "game_1.teammate_252.coordination",
            "Charles前排", "Kaiwen与Charles玩大乱斗时偏好由Charles担任前排。", "teammate_game", 1, 252,
            ["Kaiwen", "Charles", "ARAM"], [{"subject": "Charles", "predicate": "plays_role", "object": "前排"}],
        ),
        (
            "ctrl_013", "00:14:10", "和朱一闻玩时让他先开团，我再跟。", "game_1.teammate_253.coordination",
            "朱一闻先开团", "与朱一闻玩大乱斗时，由朱一闻先开团，Kaiwen随后跟进。", "teammate_game", 1, 253,
            ["Kaiwen", "朱一闻", "ARAM"], [{"subject": "朱一闻", "predicate": "initiates_before", "object": "Kaiwen"}],
        ),
        (
            "ctrl_014", "00:14:25", "毕文不喜欢太肉麻的鼓励，简单说加油就行。", "teammate_254.encouragement_style",
            "简短加油", "毕文偏好简短、不肉麻的鼓励方式。", "teammate_global", None, 254,
            ["毕文"], [{"subject": "毕文", "predicate": "prefers_encouragement", "object": "简短加油"}],
        ),
        (
            "ctrl_015", "00:15:20", "我不喜欢AI替我骂队友，默认别嘴臭。", "user.trash_talk",
            "禁止", "Kaiwen不希望AI替他辱骂队友。", "global", None, None,
            ["Kaiwen", "豆小饼"], [{"subject": "Kaiwen", "predicate": "disallows", "object": "辱骂队友"}],
        ),
        (
            "ctrl_016", "00:16:10", "如果我说救命，豆小饼应该立刻让队友保护我。", "user.rescue_protocol",
            "呼叫保护", "当Kaiwen说救命时，豆小饼应立即请求队友保护Kaiwen。", "global", None, None,
            ["Kaiwen", "豆小饼"], [{"subject": "救命", "predicate": "triggers", "object": "请求保护"}],
        ),
        (
            "ctrl_017", "00:17:10", "我名字是Kaiwen，也可以叫我凯文。", "user.name",
            "Kaiwen", "用户姓名是Kaiwen，中文别名为凯文。", "global", None, None,
            ["Kaiwen"], [{"subject": "Kaiwen", "predicate": "alias", "object": "凯文"}],
        ),
        # Game knowledge is held constant across memory policies.
        (
            "ctrl_k01", "00:06:00", "游戏知识：亚索通常需要暴击装备来放大被动收益。", "knowledge.yasuo_crit",
            "暴击装备", "亚索通常需要暴击装备来发挥被动收益。", "game_knowledge", 1, None,
            ["亚索", "ARAM"], [{"subject": "亚索", "predicate": "benefits_from", "object": "暴击装备"}],
        ),
        (
            "ctrl_k02", "00:06:05", "游戏知识：卡牌大师黄牌可以提供稳定单体控制。", "knowledge.tf_yellow_card",
            "单体控制", "卡牌大师的黄牌提供稳定的单体眩晕控制。", "game_knowledge", 1, None,
            ["卡牌大师", "ARAM"], [{"subject": "卡牌大师", "predicate": "yellow_card_provides", "object": "单体控制"}],
        ),
        (
            "ctrl_k03", "00:06:10", "游戏知识：大乱斗雪球适合远距离开团。", "knowledge.aram_snowball",
            "远距离开团", "大乱斗的雪球召唤师技能适合远距离开团。", "game_knowledge", 1, None,
            ["ARAM"], [{"subject": "雪球", "predicate": "supports", "object": "远距离开团"}],
        ),
        (
            "ctrl_k04", "00:06:15", "游戏知识：锤石灯笼可以帮助队友脱离危险。", "knowledge.thresh_lantern",
            "救援队友", "锤石的灯笼可以救援处于危险中的队友。", "game_knowledge", 1, None,
            ["锤石", "ARAM"], [{"subject": "锤石灯笼", "predicate": "supports", "object": "救援队友"}],
        ),
    ]

    events: list[Event] = []
    for row in rows:
        (
            event_id,
            hhmmss,
            text,
            slot,
            value,
            canonical,
            scope,
            game_id,
            teammate_id,
            entities,
            relations,
        ) = row
        events.append(
            Event(
                event_id=event_id,
                timestamp=iso_ts(hhmmss),
                source="controlled",
                speaker_type="user" if not slot.startswith("knowledge.") else "knowledge",
                text=text,
                event_type="knowledge" if slot.startswith("knowledge.") else "utterance",
                should_store=True,
                memory_kind="game_knowledge" if slot.startswith("knowledge.") else "memory",
                slot=slot,
                value=value,
                canonical=canonical,
                scope=scope,
                game_id=game_id,
                teammate_id=teammate_id,
                entities=entities,
                relations=relations,
            )
        )
    return events


DURABLE_MARKERS = (
    "以后",
    "默认",
    "喜欢",
    "不喜欢",
    "习惯",
    "不许叫",
    "别叫",
    "偏好",
    "我名字",
    "通常",
    "尽量",
    "改成",
    "玩时让",
    "如果",
)
EPHEMERAL_MARKERS = (
    "我死了",
    "救命啊",
    "走走",
    "没攻速",
    "杀了一个",
    "秒切",
    "输了",
    "我来了",
)
FILLERS = {"嗯", "哦", "哎", "哈哈", "呵呵", "oh", "ok", "爽", "走呀", "はい"}


def infer_store_mem0_style(event: Event) -> bool:
    if event.memory_kind == "game_knowledge":
        return True
    normalized = normalize_text(event.text)
    if len(normalized) < 5:
        return False
    if any(marker in event.text for marker in EPHEMERAL_MARKERS):
        return False
    return any(marker in event.text for marker in DURABLE_MARKERS)


def infer_store_mem9_style(event: Event) -> bool:
    if event.memory_kind == "game_knowledge":
        return True
    normalized = normalize_text(event.text)
    if len(normalized) < 4 or normalized in FILLERS:
        return False
    # Event-preserving policy: retain meaningful utterances and commands.
    return event.event_type in {"utterance", "command"}


def event_to_memory(event: Event, system: str, sequence: int) -> Memory:
    content = event.canonical if system == "mem0_style" and event.canonical else event.text
    return Memory(
        memory_id=f"{system}_{sequence:04d}",
        source_event_id=event.event_id,
        timestamp=event.timestamp,
        content=content,
        memory_kind=event.memory_kind,
        slot=event.slot,
        value=event.value,
        scope=event.scope,
        user_id=event.user_id,
        game_id=event.game_id,
        teammate_id=event.teammate_id,
        entities=event.entities or canonical_entities(content),
        relations=event.relations,
        memory_system=system,
    )


def build_memory_corpus(events: list[Event], system: str) -> list[Memory]:
    ordered = sorted(events, key=lambda e: (e.timestamp, e.event_id))
    memories: list[Memory] = []
    if system == "mem0_style":
        slots: dict[str, Memory] = {}
        for event in ordered:
            if not infer_store_mem0_style(event):
                continue
            memory = event_to_memory(event, system, len(memories) + len(slots) + 1)
            if memory.slot:
                slots[memory.slot] = memory
            else:
                memories.append(memory)
        memories.extend(slots.values())
        memories.sort(key=lambda m: (m.timestamp, m.memory_id))
        for idx, memory in enumerate(memories, 1):
            memory.memory_id = f"{system}_{idx:04d}"
        return memories

    if system == "mem9_style":
        seen: list[str] = []
        for event in ordered:
            if not infer_store_mem9_style(event):
                continue
            normalized = normalize_text(event.text)
            duplicate = any(
                SequenceMatcher(None, normalized, prior).ratio() >= 0.94 for prior in seen[-30:]
            )
            if duplicate:
                continue
            seen.append(normalized)
            memories.append(event_to_memory(event, system, len(memories) + 1))
        return memories

    raise ValueError(f"Unknown memory system: {system}")


def benchmark_queries() -> list[Query]:
    specs = [
        # Direct/global
        ("q001", "我平时希望你用什么语言回答？", "direct", None, None, [("user.language", "中文")]),
        ("q002", "游戏中回复应该长还是短？", "direct", 1, None, [("user.reply_style", "简短")]),
        ("q003", "这个AI叫什么名字？", "alias", 1, None, [("agent.nickname", "豆小饼")]),
        ("q004", "凯文的正式英文名是什么？", "alias", None, None, [("user.name", "Kaiwen")]),
        ("q005", "默认可以替我辱骂队友吗？", "policy", 1, None, [("user.trash_talk", "禁止")]),
        ("q006", "当我喊救命时AI应该做什么？", "policy", 1, None, [("user.rescue_protocol", "呼叫保护")]),
        # Temporal
        ("q007", "我目前在大乱斗最想玩什么定位？", "temporal", 1, None, [("game_1.preferred_role", "坦克")]),
        ("q008", "不要用旧偏好，我现在大乱斗玩输出还是坦克？", "temporal", 1, None, [("game_1.preferred_role", "坦克")]),
        ("q009", "我当前卡牌大师采用哪套出装？", "temporal", 1, None, [("game_1.twisted_fate_build", "攻速AD")]),
        ("q010", "卡牌出装偏好更新后是什么？", "temporal", 1, None, [("game_1.twisted_fate_build", "攻速AD")]),
        # Scope
        ("q011", "和Bill玩COD时我的定位是什么？", "scope", 2, 250, [("game_2.teammate_250.role", "狙击手")]),
        ("q012", "使命召唤里比尔应该怎样保护我？", "alias", 2, 250, [("game_2.teammate_250.coordination", "Bill守近点")]),
        ("q013", "和Bill玩无畏契约时我打什么位置？", "scope", 3, 250, [("game_3.teammate_250.role", "突破")]),
        ("q014", "同样是Bill，但在Valorant里我还用狙吗？", "scope", 3, 250, [("game_3.teammate_250.role", "突破")]),
        ("q015", "和爱丽丝玩无畏契约时我习惯什么定位？", "alias", 3, 251, [("game_3.teammate_251.role", "控场")]),
        ("q016", "Charles在大乱斗里应该站什么位置？", "scope", 1, 252, [("game_1.teammate_252.coordination", "Charles前排")]),
        ("q017", "和朱玉文玩大乱斗时谁先开团？", "alias", 1, 253, [("game_1.teammate_253.coordination", "朱一闻先开团")]),
        ("q018", "应该怎样鼓励毕文？", "scope", 1, 254, [("teammate_254.encouragement_style", "简短加油")]),
        ("q019", "朱一闻不希望AI怎么称呼他？", "real_log", 1, 253, [("teammate_253.address_preference", "不要称呼主人")]),
        ("q020", "周宇文对主人这个称呼是什么态度？", "alias", 1, 253, [("teammate_253.address_preference", "不要称呼主人")]),
        # Multi-hop
        (
            "q021", "在COD和Bill组队时，我用什么角色、Bill又怎样配合？",
            "multi_hop", 2, 250,
            [("game_2.teammate_250.role", "狙击手"), ("game_2.teammate_250.coordination", "Bill守近点")],
        ),
        (
            "q022", "比尔在使命召唤要保护近点，是因为我通常用什么武器定位？",
            "multi_hop", 2, 250,
            [("game_2.teammate_250.coordination", "Bill守近点"), ("game_2.teammate_250.role", "狙击手")],
        ),
        (
            "q023", "朱一闻开团后谁应该跟进？完整说明配合顺序。",
            "multi_hop", 1, 253,
            [("game_1.teammate_253.coordination", "朱一闻先开团")],
        ),
        # Knowledge
        ("q024", "亚索为什么经常购买暴击装备？", "knowledge", 1, None, [("knowledge.yasuo_crit", "暴击装备")]),
        ("q025", "卡牌大师哪张牌提供稳定控制？", "knowledge", 1, None, [("knowledge.tf_yellow_card", "单体控制")]),
        ("q026", "大乱斗里什么召唤师技能适合远距离开团？", "knowledge", 1, None, [("knowledge.aram_snowball", "远距离开团")]),
        ("q027", "锤石应该用什么技能救处于危险的队友？", "knowledge", 1, None, [("knowledge.thresh_lantern", "救援队友")]),
        (
            "q028", "结合我的偏好和游戏机制，我当前玩卡牌该出什么，以及黄牌有什么用？",
            "mixed", 1, None,
            [("game_1.twisted_fate_build", "攻速AD"), ("knowledge.tf_yellow_card", "单体控制")],
        ),
        (
            "q029", "我喜欢坦克时，大乱斗用什么技能方便远距离进场？",
            "mixed", 1, None,
            [("game_1.preferred_role", "坦克"), ("knowledge.aram_snowball", "远距离开团")],
        ),
        # Paraphrase/noisy queries
        ("q030", "豆小包应该用哪种语言和凯文讲话？", "noisy_alias", None, None, [("user.language", "中文")]),
        ("q031", "窦小丙回复我时别说太久，这对应什么偏好？", "noisy_alias", 1, None, [("user.reply_style", "简短")]),
        ("q032", "朱玉文和我配合的时候，是他冲还是我先冲？", "noisy_alias", 1, 253, [("game_1.teammate_253.coordination", "朱一闻先开团")]),
        ("q033", "跟比尔打枪战我架远点时，谁看近处？", "paraphrase", 2, 250, [("game_2.teammate_250.coordination", "Bill守近点")]),
        # Abstention
        ("q034", "我最喜欢的Apex英雄是谁？", "abstention", 4, None, []),
        ("q035", "Alice在COD里通常用什么角色？", "abstention", 2, 251, []),
        ("q036", "我对Dota 2的出装偏好是什么？", "abstention", 5, None, []),
        ("q037", "毕文最喜欢的食物是什么？", "abstention", None, 254, []),
        ("q038", "Charles在Valorant里打什么位置？", "abstention", 3, 252, []),
        # Additional realistic action questions
        ("q039", "有人喊救命而且我们有锤石，该如何处理？", "mixed", 1, None, [("user.rescue_protocol", "呼叫保护"), ("knowledge.thresh_lantern", "救援队友")]),
        ("q040", "给毕文加油时要不要说一大段很肉麻的话？", "policy", 1, 254, [("teammate_254.encouragement_style", "简短加油")]),
    ]
    queries: list[Query] = []
    for query_id, text, category, game_id, teammate_id, gold in specs:
        queries.append(
            Query(
                query_id=query_id,
                text=text,
                category=category,
                game_id=game_id,
                teammate_id=teammate_id,
                gold=gold,
                should_abstain=not gold,
            )
        )
    return queries


class RetrievalIndex:
    def __init__(self, memories: list[Memory], queries: list[Query]):
        self.memories = memories
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)
        texts = [m.content for m in memories]
        self.matrix = self.vectorizer.fit_transform(texts)
        self.query_entities = {q.query_id: canonical_entities(q.text) for q in queries}
        self.max_time = max(datetime.fromisoformat(m.timestamp).timestamp() for m in memories)
        self.min_time = min(datetime.fromisoformat(m.timestamp).timestamp() for m in memories)

    @staticmethod
    def base_scope_eligible(memory: Memory, query: Query) -> bool:
        if memory.user_id != query.user_id:
            return False
        if memory.game_id is not None and query.game_id != memory.game_id:
            return False
        if memory.teammate_id is not None and query.teammate_id != memory.teammate_id:
            return False
        return True

    @staticmethod
    def sag_eligible(memory: Memory, query: Query) -> bool:
        if memory.user_id != query.user_id:
            return False
        if memory.scope == "global":
            return True
        if memory.scope in {"user_game", "game_knowledge"}:
            return query.game_id == memory.game_id
        if memory.scope == "teammate_global":
            return query.teammate_id == memory.teammate_id
        if memory.scope == "teammate_game":
            return query.game_id == memory.game_id and query.teammate_id == memory.teammate_id
        # Session chatter is active only for the observed League session.
        return query.game_id == 1

    def semantic(self, query: Query) -> np.ndarray:
        qv = self.vectorizer.transform([query.text])
        return cosine_similarity(qv, self.matrix)[0]

    def entity_overlap(self, query: Query) -> np.ndarray:
        q_entities = set(self.query_entities[query.query_id])
        scores = []
        for memory in self.memories:
            m_entities = set(memory.entities)
            union = q_entities | m_entities
            scores.append(len(q_entities & m_entities) / len(union) if union else 0.0)
        return np.array(scores)

    def graph_neighbor(self, base_scores: np.ndarray) -> np.ndarray:
        entity_to_indices: defaultdict[str, list[int]] = defaultdict(list)
        for idx, memory in enumerate(self.memories):
            for entity in memory.entities:
                entity_to_indices[entity].append(idx)
        neighbor = np.zeros(len(self.memories))
        for indices in entity_to_indices.values():
            if len(indices) < 2:
                continue
            max_score = max(base_scores[idx] for idx in indices)
            for idx in indices:
                neighbor[idx] = max(neighbor[idx], max_score)
        return neighbor

    def recency(self) -> np.ndarray:
        span = max(self.max_time - self.min_time, 1.0)
        return np.array(
            [
                (datetime.fromisoformat(m.timestamp).timestamp() - self.min_time) / span
                for m in self.memories
            ]
        )

    def search(
        self,
        query: Query,
        retriever: str,
        top_k: int = 3,
        threshold: float = 0.12,
    ) -> list[tuple[Memory, float]]:
        semantic = self.semantic(query)
        entities = self.entity_overlap(query)

        if retriever == "current_rag":
            scores = semantic
            eligible = np.array([self.base_scope_eligible(m, query) for m in self.memories])
        elif retriever == "graph_rag":
            neighbor = self.graph_neighbor(semantic)
            scores = 0.62 * semantic + 0.25 * entities + 0.13 * neighbor
            eligible = np.array([self.base_scope_eligible(m, query) for m in self.memories])
        elif retriever == "sag":
            exact_scope = np.array(
                [
                    float(
                        (m.game_id is None or m.game_id == query.game_id)
                        and (m.teammate_id is None or m.teammate_id == query.teammate_id)
                    )
                    for m in self.memories
                ]
            )
            scores = (
                0.55 * semantic
                + 0.22 * entities
                + 0.18 * self.recency()
                + 0.05 * exact_scope
            )
            structural = np.array([self.sag_eligible(m, query) for m in self.memories])
            # SAG first activates a local event/entity neighborhood. Recency may
            # rank activated candidates, but it must never activate an unrelated
            # memory by itself.
            activated = (semantic >= 0.035) | (entities > 0.0)
            eligible = structural & activated
        else:
            raise ValueError(retriever)

        scores = np.where(eligible, scores, -1.0)
        order = np.argsort(-scores, kind="stable")
        results: list[tuple[Memory, float]] = []
        for idx in order:
            if scores[idx] < threshold:
                continue
            results.append((self.memories[idx], float(scores[idx])))
            if len(results) == top_k:
                break
        return results


def evaluate_query(query: Query, results: list[tuple[Memory, float]]) -> dict:
    retrieved = [memory for memory, _ in results]
    retrieved_pairs = [(m.slot, m.value) for m in retrieved if m.slot and m.value]
    gold = query.gold

    if query.should_abstain:
        answerable = float(len(results) == 0)
        recall_at_1 = recall_at_3 = mrr = answerable
        stale = 0.0
    else:
        hits = [pair in gold for pair in retrieved_pairs]
        recall_at_1 = float(bool(retrieved_pairs) and retrieved_pairs[0] in gold)
        recall_at_3 = sum(1 for pair in gold if pair in retrieved_pairs[:3]) / len(gold)
        answerable = float(all(pair in retrieved_pairs[:3] for pair in gold))
        ranks = [
            idx + 1
            for idx, pair in enumerate(retrieved_pairs)
            if pair in gold
        ]
        mrr = 1.0 / min(ranks) if ranks else 0.0
        gold_slots = {slot: value for slot, value in gold}
        stale = float(
            any(
                m.slot in gold_slots and m.value is not None and m.value != gold_slots[m.slot]
                for m in retrieved[:3]
            )
        )

    scope_leak = 0.0
    if retrieved:
        leaked = 0
        for memory in retrieved:
            if memory.game_id is not None and memory.game_id != query.game_id:
                leaked += 1
            elif memory.teammate_id is not None and memory.teammate_id != query.teammate_id:
                leaked += 1
        scope_leak = leaked / len(retrieved)

    return {
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "mrr": mrr,
        "evidence_complete": answerable,
        "stale_hit": stale,
        "scope_leakage": scope_leak,
        "returned_count": len(results),
    }


def bootstrap_ci(values: Iterable[float], seed: int = 20260724) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def memory_quality(events: list[Event], memories: list[Memory]) -> dict:
    # Evaluate the materialized *current state*, not historical update events.
    # Consolidating an obsolete value is correct and should not count as a miss.
    current_gold: dict[str, str] = {}
    for event in sorted(events, key=lambda e: (e.timestamp, e.event_id)):
        if (
            event.should_store
            and event.memory_kind == "memory"
            and event.slot
            and event.value
        ):
            current_gold[event.slot] = event.value
    stored = [m for m in memories if m.memory_kind == "memory"]
    correct_records = [
        m
        for m in stored
        if m.slot in current_gold and m.value == current_gold[m.slot]
    ]
    correct_slots = {m.slot for m in correct_records}
    precision = len(correct_records) / len(stored) if stored else 0.0
    recall = len(correct_slots) / len(current_gold) if current_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "stored_memories": len(stored),
        "gold_durable_events": len(current_gold),
        "memory_precision": precision,
        "memory_recall": recall,
        "memory_f1": f1,
        "contamination_rate": 1.0 - precision,
    }


def run(log_path: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    real_events, log_stats = parse_log(log_path)
    controlled = controlled_events()
    all_events = sorted(real_events + controlled, key=lambda e: (e.timestamp, e.event_id))
    queries = benchmark_queries()

    with (DATA_DIR / "canonical_events.jsonl").open("w", encoding="utf-8") as fh:
        for event in all_events:
            fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    with (DATA_DIR / "benchmark_queries.jsonl").open("w", encoding="utf-8") as fh:
        for query in queries:
            payload = asdict(query)
            payload["gold"] = [{"slot": s, "value": v} for s, v in query.gold]
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    (RESULTS_DIR / "log_stats.json").write_text(
        json.dumps(log_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    systems = ["mem0_style", "mem9_style"]
    retrievers = ["current_rag", "graph_rag", "sag"]
    corpora = {system: build_memory_corpus(all_events, system) for system in systems}
    for system, memories in corpora.items():
        with (DATA_DIR / f"{system}_memories.jsonl").open("w", encoding="utf-8") as fh:
            for memory in memories:
                fh.write(json.dumps(asdict(memory), ensure_ascii=False) + "\n")

    detail_rows: list[dict] = []
    metric_rows: list[dict] = []

    for system in systems:
        memories = corpora[system]
        index = RetrievalIndex(memories, queries)
        mem_quality = memory_quality(all_events, memories)
        for retriever in retrievers:
            latencies_ms: list[float] = []
            config_details: list[dict] = []
            # Repeat searches to stabilize local timing. Only the final result is scored.
            for query in queries:
                result = []
                for _ in range(30):
                    start = time.perf_counter_ns()
                    result = index.search(query, retriever)
                    latencies_ms.append((time.perf_counter_ns() - start) / 1_000_000)
                metrics = evaluate_query(query, result)
                row = {
                    "memory_system": system,
                    "retriever": retriever,
                    "query_id": query.query_id,
                    "category": query.category,
                    "query": query.text,
                    **metrics,
                    "gold": json.dumps(query.gold, ensure_ascii=False),
                    "retrieved": json.dumps(
                        [
                            {
                                "memory_id": m.memory_id,
                                "slot": m.slot,
                                "value": m.value,
                                "content": m.content,
                                "score": round(score, 6),
                            }
                            for m, score in result
                        ],
                        ensure_ascii=False,
                    ),
                }
                detail_rows.append(row)
                config_details.append(row)

            recall_values = [r["recall_at_3"] for r in config_details]
            evidence_values = [r["evidence_complete"] for r in config_details]
            ci_low, ci_high = bootstrap_ci(evidence_values)
            metric_rows.append(
                {
                    "memory_system": system,
                    "retriever": retriever,
                    **mem_quality,
                    "query_count": len(queries),
                    "recall_at_1": statistics.fmean(r["recall_at_1"] for r in config_details),
                    "recall_at_3": statistics.fmean(recall_values),
                    "mrr": statistics.fmean(r["mrr"] for r in config_details),
                    "evidence_complete_accuracy": statistics.fmean(evidence_values),
                    "evidence_accuracy_ci_low": ci_low,
                    "evidence_accuracy_ci_high": ci_high,
                    "stale_hit_rate": statistics.fmean(r["stale_hit"] for r in config_details),
                    "scope_leakage_rate": statistics.fmean(
                        r["scope_leakage"] for r in config_details
                    ),
                    "latency_p50_ms": float(np.quantile(latencies_ms, 0.50)),
                    "latency_p95_ms": float(np.quantile(latencies_ms, 0.95)),
                    "avg_returned": statistics.fmean(r["returned_count"] for r in config_details),
                }
            )

    details_df = pd.DataFrame(detail_rows)
    metrics_df = pd.DataFrame(metric_rows)
    details_df.to_csv(RESULTS_DIR / "query_results.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    category_df = (
        details_df.groupby(["memory_system", "retriever", "category"], as_index=False)
        .agg(
            query_count=("query_id", "count"),
            recall_at_3=("recall_at_3", "mean"),
            evidence_complete_accuracy=("evidence_complete", "mean"),
            stale_hit_rate=("stale_hit", "mean"),
        )
    )
    category_df.to_csv(RESULTS_DIR / "category_metrics.csv", index=False)

    failure_df = details_df[details_df["evidence_complete"] < 1.0].copy()
    failure_df.to_csv(RESULTS_DIR / "failures.csv", index=False)

    # Paired query-level improvements for architectural interpretation.
    pivot = details_df.pivot_table(
        index=["memory_system", "query_id"],
        columns="retriever",
        values="evidence_complete",
    ).reset_index()
    pairwise_rows = []
    for system in systems:
        subset = pivot[pivot["memory_system"] == system]
        for challenger in ["graph_rag", "sag"]:
            deltas = subset[challenger] - subset["current_rag"]
            low, high = bootstrap_ci(deltas)
            pairwise_rows.append(
                {
                    "memory_system": system,
                    "comparison": f"{challenger} - current_rag",
                    "paired_accuracy_delta": float(deltas.mean()),
                    "delta_ci_low": low,
                    "delta_ci_high": high,
                    "wins": int((deltas > 0).sum()),
                    "losses": int((deltas < 0).sum()),
                    "ties": int((deltas == 0).sum()),
                }
            )
    for retriever in retrievers:
        subset = details_df[details_df["retriever"] == retriever].pivot(
            index="query_id",
            columns="memory_system",
            values="evidence_complete",
        )
        deltas = subset["mem0_style"] - subset["mem9_style"]
        low, high = bootstrap_ci(deltas)
        pairwise_rows.append(
            {
                "memory_system": "memory_policy_comparison",
                "comparison": f"mem0_style - mem9_style under {retriever}",
                "paired_accuracy_delta": float(deltas.mean()),
                "delta_ci_low": low,
                "delta_ci_high": high,
                "wins": int((deltas > 0).sum()),
                "losses": int((deltas < 0).sum()),
                "ties": int((deltas == 0).sum()),
            }
        )
    pd.DataFrame(pairwise_rows).to_csv(RESULTS_DIR / "paired_comparisons.csv", index=False)

    # Compact mentor-report chart.
    plot_df = metrics_df.copy()
    plot_df["configuration"] = (
        plot_df["memory_system"].str.replace("_style", "", regex=False)
        + "\n"
        + plot_df["retriever"].str.replace("_rag", "", regex=False)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#2563eb", "#7c3aed", "#059669", "#60a5fa", "#a78bfa", "#34d399"]
    axes[0].bar(
        plot_df["configuration"],
        plot_df["evidence_complete_accuracy"] * 100,
        color=colors,
    )
    axes[0].set_title("Evidence-complete accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis="x", rotation=25)
    for idx, value in enumerate(plot_df["evidence_complete_accuracy"] * 100):
        axes[0].text(idx, value + 1.2, f"{value:.1f}", ha="center", fontsize=8)

    axes[1].bar(
        plot_df["configuration"],
        plot_df["latency_p95_ms"],
        color=colors,
    )
    axes[1].set_title("Local retrieval p95 latency")
    axes[1].set_ylabel("Milliseconds")
    axes[1].tick_params(axis="x", rotation=25)
    for idx, value in enumerate(plot_df["latency_p95_ms"]):
        axes[1].text(idx, value + max(plot_df["latency_p95_ms"]) * 0.02, f"{value:.2f}", ha="center", fontsize=8)
    fig.suptitle("Voice-agent memory × retrieval pilot (controlled local proxies)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "benchmark_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(metrics_df.to_string(index=False))
    print(f"\nWrote benchmark artifacts to {ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    run(args.log.resolve())


if __name__ == "__main__":
    main()
