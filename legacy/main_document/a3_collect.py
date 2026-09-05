#!/usr/bin/env python3
"""a3_collect.py — A3 selective selection collection (v6 protocol).
 
Phase 1: Student (SFT_v6, P_S) rolls out on the SAME 50 games as A1 round 1,
         recording every turn's state + admissible actions.
Phase 2: transformers batch scoring -> admissible-action entropy H_t per turn.
Phase 3: per game, select the top-3 entropy turns -> Teacher (P_T, thinking) gives
         N=1 corrections on the exact same states (150 budget, same as random A1).
"""
from __future__ import annotations
 
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
 
import yaml
from openai import OpenAI
from transformers import AutoTokenizer
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# agent_harness lives on /cfs (s3fs) — read-only is fine
sys.path.insert(0, '/cfs/data/private/yangchunyu/ld/agent_opd_route_a')
from agent_harness.parser import parse_and_canonicalize  # noqa: E402
from agent_harness.context import TaskPreservingTruncator  # noqa: E402
 
STUDENT_SYSTEM_PROMPT = """You are an ALFWorld household agent.
 
Complete the given household task by interacting with the environment one step at a time.
 
At each turn, use the task description, interaction history, current observation, and current admissible actions to choose the next action.
 
Return exactly one action from the current admissible actions in this format:
Action: <command>
 
Use the action exactly as written in the admissible action list. Do not change object names, object numbers, or command syntax.
 
Do not output reasoning, explanations, multiple actions, predicted observations, or future turns."""
 
TEACHER_SYSTEM_PROMPT = """You are an expert ALFWorld household agent.
 
Complete the given household task by interacting with the environment one step at a time.
 
At each turn, use the task description, interaction history, current observation, and current admissible actions to determine the next action.
 
Base your decision only on objects, receptacles, locations, and state information supported by the task and the actual interaction history.
 
Keep the task's target object, its current location, carried objects, and relevant state changes such as cleaned, heated, or cooled consistent with the observed environment.
 
Do not invent object identities, object numbers, locations, carried objects, or state changes.
 
Choose exactly one action from the current admissible actions.
 
Return exactly:
Action: <command>
 
Use the action exactly as written in the admissible action list.
 
Do not output reasoning, explanations, multiple actions, predicted observations, or future turns."""
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A3 selective (top-entropy) Teacher correction collection")
    p.add_argument("--env-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--only-games-jsonl", required=True,
                   help="jsonl (e.g. A1 run1 corrections) whose game_ids are the EXACT games to replay")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m-turns", type=int, default=3)
    p.add_argument("--student-model", default=None)
    p.add_argument("--student-base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--student-temperature", type=float, default=0.0)
    p.add_argument("--student-max-tokens", type=int, default=256)
    p.add_argument("--teacher-api-key", default=None)
    p.add_argument("--teacher-key-file", default="/root/.deepseek_key")
    p.add_argument("--teacher-model", default="deepseek-v4-flash")
    p.add_argument("--teacher-max-tokens", type=int, default=3000)
    p.add_argument("--teacher-max-retries", type=int, default=2)
    p.add_argument("--tokenizer-path", default="/cfs/data/private/yangchunyu/ld/models/qwen3-4b-sft")
    p.add_argument("--context-max-tokens", type=int, default=4096)
    p.add_argument("--context-reserve-tokens", type=int, default=256)
    # entropy scoring model (transformers): base + sft-v6 LoRA = the SFT_v6 student
    p.add_argument("--score-base", default="/cfs/data/private/zhangsl/Model/Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--score-lora", default="/root/data/alfworld/checkpoints/sft_v6/global_step_542")
    p.add_argument("--phase", choices=["rollout", "score"], default="rollout",
                   help="rollout: student plays games, saves episodes.jsonl; "
                        "score: load episodes, entropy-rank, teacher-correct")
    p.add_argument("--episodes-input", default=None,
                   help="score phase: episodes jsonl to load (default: <output>/episodes.jsonl)")
    p.add_argument("--gpu", type=int, default=0, help="score phase: cuda device index")
    p.add_argument("--selection", choices=["top", "bounded", "mixed"], default="top",
                   help="top: per-game top-M entropy (A3); bounded: per-game entropy "
                        "percentile in [bounded_lo, bounded_hi] then top-M (A4); "
                        "mixed: 1 random + 2 bounded[lo,hi] per game (A5)")
    p.add_argument("--bounded-lo", type=float, default=0.6)
    p.add_argument("--bounded-hi", type=float, default=0.9)
    return p.parse_args()
 
 
def read_secret(cli_value, key_file, env_name):
    if cli_value and cli_value != "EMPTY":
        return cli_value.strip()
    env_value = os.getenv(env_name)
    if env_value:
        return env_value.strip()
    if key_file and Path(key_file).exists():
        raw = Path(key_file).read_text(encoding="utf-8").strip()
        if "=" in raw and not raw.startswith("sk-"):
            raw = raw.split("=", 1)[1]
        return raw.strip()
    raise RuntimeError(f"Missing API key: {env_name} or key file {key_file}")
 
 
def task_description(initial_observation: str) -> str:
    parts = str(initial_observation).split("\n\n")
    return "\n".join(parts[1:]).strip() if len(parts) > 1 else str(initial_observation).strip()
 
 
def build_one_game_env(env_config, game: str):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    method = env_config["general"]["training_method"]
    if method == "dagger":
        max_episode_steps = env_config["dagger"]["training"]["max_nb_steps_per_episode"]
    elif method == "dqn":
        max_episode_steps = env_config["rl"]["training"]["max_nb_steps_per_episode"]
    else:
        raise ValueError(f"Unsupported training_method={method!r}")
    env_id = textworld.gym.register_games(
        [game], request_infos, batch_size=1, auto_reset=False, asynchronous=False,
        max_episode_steps=max_episode_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
    )
    return textworld.gym.make(env_id)
 
 
def unwrap_info(info, index=0):
    out = {}
    for key, value in info.items():
        try:
            out[key] = value[index]
        except Exception:
            out[key] = value
    return out
 
 
def admissible_block(admissible) -> str:
    return "\n".join(f"- {a}" for a in admissible)
 
 
def first_user_message(desc: str, observation: str, admissible) -> str:
    return (f"Task:\n{desc}\n\nObservation:\n{observation.strip()}\n\nAdmissible actions:\n"
            + admissible_block(admissible))
 
 
def turn_user_message(observation: str, admissible) -> str:
    return (f"Observation:\n{observation.strip()}\n\nAdmissible actions:\n"
            + admissible_block(admissible))
 
 
def classify_failure(raw, parsed):
    if not str(raw or "").strip():
        return "empty_content"
    if not parsed.had_action_marker:
        return "no_action_marker"
    if not parsed.valid:
        return "not_admissible_after_canonicalization"
    return None
 
 
def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    env_cfg = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
 
    # exact game list: the SAME games as A1 round 1
    games = []
    for line in Path(args.only_games_jsonl).open(encoding="utf-8"):
        if line.strip():
            g = json.loads(line).get("gamefile")
            if g and g not in games:
                games.append(g)
    games = games[: 10000]  # keep all (50 from run1)
    rng.shuffle(games)
    print(json.dumps({"games_selected": len(games)}, ensure_ascii=False), flush=True)
 
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / "episodes.jsonl"
    out_path = out_dir / "corrections.jsonl"
    summary_path = out_dir / "summary.json"
 
    # ---------------- Phase 1 (rollout): Student plays, record all turns ----------------
    if args.phase == "rollout":
        assert args.student_model, "--student-model required for rollout"
        student = OpenAI(api_key="EMPTY", base_url=args.student_base_url)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        truncator = TaskPreservingTruncator(tokenizer, max_context_tokens=args.context_max_tokens,
                                            reserve_tokens=args.context_reserve_tokens,
                                            enable_thinking=False)
        episodes = []
        for ep_idx, game in enumerate(games):
            env = build_one_game_env(env_cfg, game)
            try:
                observations, info = env.reset()
                info0 = unwrap_info(info)
                desc = task_description(str(observations[0]))
                gf = str(info0.get("extra.gamefile") or game)
                task_type = gf.split("/")[-3] if "/" in gf else "?"
                full_history = [
                    {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                    {"role": "user", "content": first_user_message(desc, desc,
                                                                   list(info0.get("admissible_commands") or []))},
                ]
                cur_obs = desc
                won = False
                done = False
                turns = []
                for step in range(args.max_steps):
                    admissible = list(info0.get("admissible_commands") or [])
                    obs_content = turn_user_message(cur_obs, admissible)
                    query, _ = truncator.truncate(full_history)
                    if step > 0:
                        query.append({"role": "user", "content": obs_content})
                    resp = student.chat.completions.create(
                        model=args.student_model, messages=query,
                        temperature=args.student_temperature, max_tokens=args.student_max_tokens)
                    content = (resp.choices[0].message.content or "") if resp.choices else ""
                    parsed = parse_and_canonicalize(content, admissible, marker_strategy="first")
                    action = parsed.canonical_action if parsed.valid else "look"
                    turns.append({
                        "turn_index": step,
                        "query_messages": query,
                        "admissible_actions": admissible,
                        "student_raw": content,
                        "student_action": action,
                        "student_valid": bool(parsed.valid),
                    })
                    full_history.append({"role": "assistant", "content": f"Action: {action}"})
                    observations, rewards, dones, infos = env.step([action])
                    cur_obs = str(observations[0])
                    info0 = unwrap_info(infos)
                    done = bool(dones[0])
                    won = bool(info0.get("won", False))
                    if done:
                        break
                    full_history.append({"role": "user", "content": obs_content})
                ep_rec = {"gamefile": gf, "task_type": task_type, "won": bool(won), "turns": turns}
                episodes.append(ep_rec)
                with episodes_path.open("a", encoding="utf-8") as ef:
                    ef.write(json.dumps(ep_rec, ensure_ascii=False) + "\n")
                print(json.dumps({"episode": ep_idx + 1, "games": len(games), "won": bool(won),
                                  "steps": len(turns), "task_type": task_type}, ensure_ascii=False), flush=True)
            finally:
                try:
                    env.close()
                except Exception:
                    pass
        print(json.dumps({"phase": "rollout-done", "episodes": len(episodes), "file": str(episodes_path)},
                         ensure_ascii=False), flush=True)
        return
 
    # ---------------- Phase 2 (score): load episodes, entropy, select, correct ----------------
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    torch.cuda.set_device(args.gpu)
    ep_in = args.episodes_input or str(episodes_path)
    episodes = [json.loads(l) for l in Path(ep_in).open(encoding="utf-8")]
    print(f"loaded {len(episodes)} episodes from {ep_in}", flush=True)
    key = read_secret(args.teacher_api_key, args.teacher_key_file, "DEEPSEEK_API_KEY")
    teacher = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    teacher_extra = {"thinking": {"type": "enabled", "effort": "high"}}
 
    # ---------------- Phase 2: admissible-action entropy via transformers ----------------
    scorer = AutoModelForCausalLM.from_pretrained(args.score_base, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16,
                                                  attn_implementation="flash_attention_2").cuda().eval()
    scorer = PeftModel.from_pretrained(scorer, args.score_lora).eval()
    tok2 = AutoTokenizer.from_pretrained(args.score_base, trust_remote_code=True)
    tok2.padding_side = 'left'  # required by flash attention 2 batched scoring
 
    def action_logprobs_batch(messages_list, action_lists, batch_size=1):
        """Return per-turn list of {action: logp} via teacher-forced batch forward.
        Uses FULL rollout context (no re-truncation: entropy must match the actual
        Student policy context), hidden_states + chunked lm_head for memory."""
        results = [None] * len(messages_list)
        seqs = []
        for ti, (msgs, acts) in enumerate(zip(messages_list, action_lists)):
            for a in acts:
                seqs.append((ti, a))
        lm_head = scorer.get_output_embeddings()
        with torch.no_grad():
            for start in range(0, len(seqs), batch_size):
                chunk = seqs[start:start + batch_size]
                texts = []
                metas = []
                for ti, a in chunk:
                    msgs_full = messages_list[ti] + [{"role": "assistant", "content": f"Action: {a}"}]
                    texts.append(tok2.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False))
                    metas.append((ti, a))
                enc_all = tok2(texts)  # BatchEncoding, input_ids = list of lists (FULL context)
                ids_list = [torch.tensor(row) for row in enc_all["input_ids"]]
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [x.flip(0) for x in ids_list], batch_first=True, padding_value=tok2.pad_token_id).flip(1)
                attn = (input_ids != tok2.pad_token_id).long()
                input_ids = input_ids.to("cuda")
                attn = attn.to("cuda")
                out = scorer(input_ids=input_ids, attention_mask=attn,
                             output_hidden_states=True)
                hidden = out.hidden_states[-1][:, :-1].to(torch.bfloat16)  # (B, T-1, H)
                targets = input_ids[:, 1:]  # (B, T-1)
                pad_id = tok2.pad_token_id or 0
                valid = (targets != pad_id).float()  # left-padding mask
                B, T, H = hidden.shape
                # chunked lm_head over the seq dim
                seq_logp = torch.zeros(B, device="cuda", dtype=torch.float32)
                CHUNK = 256
                for s in range(0, T, CHUNK):
                    h = hidden[:, s:s + CHUNK]  # (B, C, H)
                    lg = lm_head(h).to(torch.bfloat16)  # (B, C, V) bf16
                    lsm = torch.nn.functional.log_softmax(lg, dim=-1)
                    tgt = targets[:, s:s + CHUNK]
                    v = valid[:, s:s + CHUNK]
                    per = lsm.gather(2, tgt.unsqueeze(-1)).squeeze(-1)  # (B, C)
                    seq_logp += (per * v).sum(dim=1)
                for bi, (ti, a) in enumerate(metas):
                    results[ti] = results[ti] or {}
                    results[ti][a] = seq_logp[bi].item()
        return results
 
    print("scoring entropy over all turns...", flush=True)
    for ep in episodes:
        msgs_list = [t["query_messages"] for t in ep["turns"]]
        act_list = [t["admissible_actions"] for t in ep["turns"]]
        logp_dicts = action_logprobs_batch(msgs_list, act_list)
        for t, lp in zip(ep["turns"], logp_dicts):
            # softmax over admissible (sum-of-token-logprob scores, as defined)
            acts = t["admissible_actions"]
            vals = torch.tensor([lp.get(a, -100.0) for a in acts], dtype=torch.float32)
            vals = vals - vals.max()
            probs = torch.softmax(vals, dim=0)
            entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
            # FROZEN metric: normalized entropy H / log|A_t| (action-set size invariant)
            n_act = max(len(acts), 2)
            t["entropy"] = entropy / math.log(n_act)
            t["action_logprobs"] = {a: round(lp.get(a, -100.0), 3) for a in acts}
    # persist full-turn entropy (per-shard file to avoid overwrite races; A5/B reuse)
    ep_in_path = Path(args.episodes_input or str(episodes_path))
    ep_out = ep_in_path.with_name(ep_in_path.stem + "_with_entropy.jsonl")
    with ep_out.open("w", encoding="utf-8") as ef:
        for ep in episodes:
            ef.write(json.dumps(ep, ensure_ascii=False) + "\n")
    print(f"persisted full entropy -> {ep_out}", flush=True)
    del scorer
    torch.cuda.empty_cache()
 
    # ---------------- Phase 3: top-3 entropy turns -> Teacher corrections ----------------
    stats = {"episodes": 0, "student_wins": 0, "corrections_total": 0, "corrections_valid": 0,
             "disagreement": 0, "teacher_retries": 0}
    all_recs = []
    with out_path.open("w", encoding="utf-8") as f:
        for ep_idx, ep in enumerate(episodes):
            stats["episodes"] += 1
            stats["student_wins"] += int(ep["won"])
            # top-M by entropy (A3) / bounded percentile (A4) / mixed 1R+2B (A5)
            ranked = sorted(enumerate(ep["turns"]), key=lambda x: x[1]["entropy"], reverse=True)
            n_turns = len(ep["turns"])
            if args.selection == "bounded":
                # per-game entropy percentile band [lo, hi] (low->high entropy);
                # ranked is DESCENDING, so band = ranked[(1-hi)*n : (1-lo)*n]
                start_i = int(n_turns * (1.0 - args.bounded_hi))
                end_i = int(n_turns * (1.0 - args.bounded_lo))
                band = ranked[start_i:end_i]
                # within band, keep highest entropy, capped at M
                band = sorted(band, key=lambda x: x[1]["entropy"], reverse=True)[: args.m_turns]
                selected = band
            elif args.selection == "mixed":
                # A5: 1 random turn + (M-1) bounded[lo,hi] top-entropy turns
                start_i = int(n_turns * (1.0 - args.bounded_hi))
                end_i = int(n_turns * (1.0 - args.bounded_lo))
                band = ranked[start_i:end_i]
                band = sorted(band, key=lambda x: x[1]["entropy"], reverse=True)
                sel_bounded = band[: args.m_turns - 1]
                pool = [t for t in enumerate(ep["turns"]) if t[0] not in {i for i, _ in sel_bounded}]
                sel_random = rng.sample(pool, 1)
                selected = sel_bounded + sel_random
            else:
                selected = ranked[: min(args.m_turns, len(ranked))]
            for turn_idx, t in selected:
                raw = ""
                parsed = None
                for attempt in range(args.teacher_max_retries + 1):
                    resp = teacher.chat.completions.create(
                        model=args.teacher_model, messages=t["query_messages"],
                        temperature=0.0, max_tokens=args.teacher_max_tokens,
                        extra_body=teacher_extra)
                    c = resp.choices[0]
                    raw = (c.message.content or "") if c else ""
                    parsed = parse_and_canonicalize(raw, t["admissible_actions"], marker_strategy="first")
                    reason = classify_failure(raw, parsed)
                    if reason in (None, "not_admissible_after_canonicalization"):
                        break
                    stats["teacher_retries"] += 1
                t_action = parsed.canonical_action if (parsed and parsed.valid) else "look"
                valid = bool(parsed and parsed.valid)
                disagree = valid and (t_action != t["student_action"])
                stats["corrections_total"] += 1
                stats["corrections_valid"] += int(valid)
                stats["disagreement"] += int(disagree)
                rec = {
                    "episode_index": ep_idx,
                    "gamefile": ep["gamefile"],
                    "task_type": ep["task_type"],
                    "student_won": ep["won"],
                    "turn_index": turn_idx,
                    "turn_depth": turn_idx,
                    "entropy": round(t["entropy"], 4),
                    "query_messages": t["query_messages"],
                    "admissible_actions": t["admissible_actions"],
                    "student_action": t["student_action"],
                    "student_valid": t["student_valid"],
                    "teacher_raw": raw,
                    "teacher_action": t_action,
                    "teacher_valid": valid,
                    "disagreement": disagree,
                }
                all_recs.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
 
    summary = {
        "protocol": "A3-selective-v6",
        "selection": "top-entropy M=3",
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "same_games_as": str(Path(args.only_games_jsonl).name),
        "stats": stats,
        "disagreement_rate": stats["disagreement"] / max(stats["corrections_total"], 1),
        "correction_valid_rate": stats["corrections_valid"] / max(stats["corrections_total"], 1),
        "student_success_rate": stats["student_wins"] / max(stats["episodes"], 1),
        "avg_selected_entropy": round(sum(r["entropy"] for r in all_recs) / max(len(all_recs), 1), 4),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
 
 
if __name__ == "__main__":
    main()
