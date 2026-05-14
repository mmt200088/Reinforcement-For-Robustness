from __future__ import annotations

import ast
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
ACTION_SPACE = ROOT / "blb_stage2_rl" / "action_space.py"
BRIDGE = ROOT / "rescale_optimizer_bridge.py"
STATIC_SKELETONS = ROOT / "Rescale_optimizer" / "configs" / "mrpc" / "static_skeletons_mrpc.json"

NUM_LAYERS = 12
K_LEVELS = (8, 9, 11, 13, 10, 12)
LEVELS_BY_KIND = {"F": 5, "W": 5, "M": 3, "S": 3, "R": 4, "K": len(K_LEVELS)}
KIND_CN = {
    "F": "fresh 噪声",
    "W": "权重 encode 噪声",
    "M": "mask/矩阵 encode 噪声",
    "S": "标量 encode 噪声",
    "R": "rescale 后结果噪声",
    "K": "PPTI 截断 k",
}
BLOCK_CN = {
    1: "Block 1：上一层 FFN2 输出到 layernorm 方差链",
    2: "Block 2：layernorm 后 Q/K/V 与 QK^T 前处理",
    3: "Block 3：softmax 多项式指数近似",
    4: "Block 4：softmax 与 V、Wo、attention 后 layernorm",
    5: "Block 5：FFN1、GELU 多项式、FFN 前 layernorm",
}

CFG_TO_RL_BASE: Dict[int, Dict[str, str]] = {
    1: {
        "gelu_out_fresh": "gelu_out_sf",
        "wffn2_result_rescale": "wffn2_rescale_sf",
        "mean_result_rescale": "mean_rescale_sf",
        "square_result_rescale": "square_rescale_sf",
        "var_result_rescale": "var_rescale_sf",
    },
    2: {
        "inv_std_fresh": "inv_std_fresh_sf",
        "x_centered_fresh": "x_centered_fresh_sf",
        "normalize_result_rescale": "normalize_rescale_sf",
        "gamma_result_rescale": "gamma_rescale_sf",
        "wk_result_rescale": "wk_rescale_sf",
        "wq_result_rescale": "wq_rescale_sf",
        "wv_result_rescale": "wv_rescale_sf",
        "kt_mask1_result_rescale": "kt_mask1_rescale_sf",
        "kt_mask2_result_rescale": "kt_mask2_rescale_sf",
        "q_mask1_result_rescale": "q_mask1_rescale_sf",
        "q_mask2_result_rescale": "q_mask2_rescale_sf",
        "qkt_matmul_result_rescale": "qkt_matmul_rescale_sf",
        "qkt_merge_mask_result_rescale": "qkt_merge_mask_rescale_sf",
    },
    3: {
        "x_fresh": "x_fresh_sf",
        "x_inv_2n_result_rescale": "x_inv_2n_rescale_sf",
    },
    4: {
        "softmax_out_fresh": "softmax_out_fresh_sf",
        "v_fresh": "v_fresh_sf",
        "softmax_out_mask_rescale": "softmax_out_mask_rescale_sf",
        "v_mask_rescale": "v_mask_rescale_sf",
        "softmax_v_matmul_rescale": "softmax_v_matmul_rescale_sf",
        "softmax_v_mask_rescale": "softmax_v_mask_rescale_sf",
        "wo_result_rescale": "wo_rescale_sf",
        "ln_mean_result_rescale": "ln_mean_rescale_sf",
        "ln_square_result_rescale": "ln_square_rescale_sf",
        "ln_var_result_rescale": "ln_var_rescale_sf",
    },
    5: {
        "inv_std_fresh": "inv_std_fresh_sf",
        "x_centered_fresh": "x_centered_fresh_sf",
        "normalize_result_rescale": "normalize_rescale_sf",
        "gamma_result_rescale": "gamma_rescale_sf",
        "wffn1_result_rescale": "wffn1_rescale_sf",
    },
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def tag(name: str, content: str, **attrs: str) -> str:
    attr = "".join(f' {k.rstrip("_")}="{esc(v)}"' for k, v in attrs.items() if v is not None)
    return f"<{name}{attr}>{content}</{name}>"


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], cls: str = "") -> str:
    head = "".join(tag("th", esc(h)) for h in headers)
    body_rows = []
    for row in rows:
        body_rows.append(tag("tr", "".join(tag("td", str(c)) for c in row)))
    return f'<table class="{esc(cls)}"><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def page(title: str, lead: str, body: str) -> str:
    css = """
    :root { --ink:#17202a; --muted:#5d6d7e; --line:#d8dee9; --bg:#f7f9fb; --panel:#fff; --blue:#1f6feb; --green:#1a7f37; --amber:#9a6700; --red:#b42318; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif; }
    main { max-width:1280px; margin:0 auto; padding:32px 22px 56px; }
    h1 { margin:0 0 8px; font-size:28px; line-height:1.2; letter-spacing:0; }
    h2 { margin:30px 0 12px; font-size:21px; letter-spacing:0; }
    h3 { margin:20px 0 8px; font-size:16px; letter-spacing:0; }
    p { margin:8px 0; }
    code { background:#edf2f7; border:1px solid #d8dee9; border-radius:4px; padding:1px 5px; font-family:Consolas,Menlo,monospace; font-size:.92em; }
    table { width:100%; border-collapse:collapse; background:var(--panel); margin:12px 0 20px; }
    th, td { border:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }
    th { background:#edf2f7; font-weight:650; position:sticky; top:0; z-index:1; }
    .lead { color:var(--muted); max-width:980px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px 16px; margin:14px 0; }
    .ok { border-left:4px solid var(--green); }
    .warn { border-left:4px solid var(--amber); }
    .risk { border-left:4px solid var(--red); }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:10px; margin:12px 0 18px; }
    .metric { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:12px; }
    .metric strong { display:block; color:var(--blue); font-size:20px; }
    .small { color:var(--muted); font-size:12px; }
    .nowrap { white-space:nowrap; }
    .formula { background:#fff8e1; border:1px solid #ead28b; border-radius:6px; padding:12px 14px; margin:12px 0; }
    math { display:block; margin:6px 0; font-size:1.05rem; overflow-x:auto; }
    @media (max-width:700px) { main { padding:24px 14px 42px; } h1 { font-size:23px; } table { font-size:12px; } }
    """
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>{css}</style>
</head>
<body>
<main>
  <header>
    <h1>{esc(title)}</h1>
    <p class="lead">{esc(lead)}</p>
    <p class="small">生成日期：2026-05-13；数据源：当前工作树中的 action_space.py、rescale_optimizer_bridge.py 与 static_skeletons_mrpc.json。</p>
  </header>
{body}
</main>
</body>
</html>
"""


def parse_python(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def get_assign(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name and node.value is not None:
                return node.value
    raise KeyError(name)


def literal(node: ast.AST) -> Any:
    return ast.literal_eval(node)


def parse_block_specs(tree: ast.Module) -> Dict[int, List[Tuple[str, str, int]]]:
    out: Dict[int, List[Tuple[str, str, int]]] = {}
    for block in range(1, 6):
        value = get_assign(tree, f"_BLOCK{block}_FIELDS")
        if not isinstance(value, ast.Call):
            raise TypeError(f"_BLOCK{block}_FIELDS is not a call")
        fields_node = None
        for kw in value.keywords:
            if kw.arg == "fields":
                fields_node = kw.value
        if fields_node is None:
            raise KeyError(f"_BLOCK{block}_FIELDS.fields")
        out[block] = [(str(a), str(b), int(c)) for a, b, c in literal(fields_node)]
    return out


def parse_node_map(tree: ast.Module) -> Dict[int, Dict[str, str]]:
    value = get_assign(tree, "_BLOCK_NODE_NAME_BY_FIELD")
    raw = literal(value)
    return {int(k): {str(a): str(b) for a, b in v.items()} for k, v in raw.items()}


def parse_t_new_map(tree: ast.Module) -> Dict[str, List[Tuple[str, Optional[int]]]]:
    value = get_assign(tree, "DEFAULT_CFG_TO_T_NEW_MAP")
    if not isinstance(value, ast.Dict):
        raise TypeError("DEFAULT_CFG_TO_T_NEW_MAP is not a dict")
    out: Dict[str, List[Tuple[str, Optional[int]]]] = {}
    for key_node, val_node in zip(value.keys, value.values):
        graph_key = str(literal(key_node))
        entries: List[Tuple[str, Optional[int]]] = []
        if not isinstance(val_node, (ast.Tuple, ast.List)):
            raise TypeError(f"{graph_key} map is not tuple/list")
        for elt in val_node.elts:
            if not isinstance(elt, ast.Call):
                raise TypeError(f"{graph_key} entry is not _SkelEntry call")
            cfg_field = str(literal(elt.args[0]))
            tuple_index = None
            if len(elt.args) >= 2:
                tuple_index = int(literal(elt.args[1]))
            entries.append((cfg_field, tuple_index))
        out[graph_key] = entries
    return out


def graph_block(graph_key: str) -> int:
    m = re.match(r"block([1-5])", graph_key)
    if not m:
        raise ValueError(graph_key)
    return int(m.group(1))


def graph_degree(graph_key: str) -> Optional[int]:
    m = re.search(r"_n(\d+)$", graph_key)
    return int(m.group(1)) if m else None


def cfg_entry_to_rl_field(block: int, graph_key: str, cfg_field: str, tuple_index: Optional[int]) -> Tuple[Optional[str], str]:
    if cfg_field == "square_rescales":
        idx = int(tuple_index or 0)
        note = "softmax 平方 rescale 槽"
        if graph_key in {"block3_exp_n5", "block3_exp_n6"} and idx == 3:
            note += "；degree>4 时复用最后一个 RL square 槽"
        return f"square_rescale_sf_{idx}", note
    if cfg_field == "gelu_power_rescales":
        idx = int(tuple_index or 0)
        return f"gelu_power_rescale_sf_{idx}", "GELU power rescale 槽"
    if cfg_field == "gelu_coeff_mul_rescales":
        deg = graph_degree(graph_key)
        idx = int(tuple_index) if tuple_index is not None and tuple_index >= 0 else int(deg or 1) - 1
        return f"gelu_coeff_mul_rescale_sf_{idx}", "GELU 系数乘法 rescale 槽；-1 表示当前 degree 的最后一个系数项"
    field = CFG_TO_RL_BASE.get(block, {}).get(cfg_field)
    return field, "直接字段映射" if field else "未找到 cfg→RL 字段映射"


def level_values(kind: str, default_max: int) -> str:
    if kind == "K":
        return ", ".join(f"{i}→{v}" for i, v in enumerate(K_LEVELS))
    if kind == "R":
        return "0→关闭；" + "；".join(
            f"{i}→{default_max - 2 * (LEVELS_BY_KIND[kind] - 1 - i)}"
            for i in range(1, LEVELS_BY_KIND[kind])
        )
    levels = LEVELS_BY_KIND[kind]
    return "；".join(f"{i}→{default_max - 2 * (levels - 1 - i)}" for i in range(levels))


def local_fields(block_specs: Mapping[int, List[Tuple[str, str, int]]]) -> List[Tuple[int, str, str, int, int]]:
    rows: List[Tuple[int, str, str, int, int]] = []
    offset = 0
    for block in range(1, 6):
        for field, kind, default_max in block_specs[block]:
            rows.append((offset, block, field, kind, default_max))
            offset += 1
    return rows


def static_entries() -> Dict[str, dict]:
    data = json.loads(STATIC_SKELETONS.read_text(encoding="utf-8"))
    return {str(e["config_name"]): e for e in data.get("results", []) if e.get("success")}


def t_base_rows(entry: Mapping[str, Any]) -> List[dict]:
    cut_by_i = {int(row["i"]): row for row in entry.get("cut_point_sf", [])}
    rows = []
    for i in [int(x) for x in entry.get("skeleton", [])]:
        row = cut_by_i.get(i)
        if not row:
            continue
        sf = row.get("sf_post", row.get("sf"))
        if sf is None:
            continue
        rows.append({
            "i": i,
            "name": str(row.get("name", "")),
            "type": str(row.get("type", "")),
            "baseline_sf": int(sf),
            "source": "sf_post" if "sf_post" in row else "sf",
        })
    return rows


def build_model(block_specs, node_map, t_map, static):
    fields = local_fields(block_specs)
    by_field = {(block, field): (offset, kind, default_max) for offset, block, field, kind, default_max in fields}
    t_rows = []
    audit_errors: List[str] = []
    audit_warnings: List[str] = []
    for graph_key, entries in sorted(t_map.items()):
        block = graph_block(graph_key)
        static_entry = static.get(graph_key)
        if not static_entry:
            audit_errors.append(f"{graph_key}: DEFAULT_CFG_TO_T_NEW_MAP 有映射，但 static_skeletons_mrpc.json 缺少该 graph")
            base = []
        else:
            base = t_base_rows(static_entry)
        if len(entries) != len(base):
            audit_errors.append(f"{graph_key}: t_new 映射长度 {len(entries)} != static baseline t 长度 {len(base)}")
        for idx, (cfg_field, tuple_index) in enumerate(entries):
            rl_field, note = cfg_entry_to_rl_field(block, graph_key, cfg_field, tuple_index)
            base_row = base[idx] if idx < len(base) else {}
            if rl_field is None:
                audit_errors.append(f"{graph_key}.t_new[{idx}]: cfg 字段 {cfg_field} 无法映射到 RL 字段")
            elif (block, rl_field) not in by_field:
                audit_errors.append(f"{graph_key}.t_new[{idx}]: RL 字段 block{block}.{rl_field} 不存在于 _BLOCK_SPECS")
            offset, kind, default_max = by_field.get((block, rl_field or ""), (-1, "?", -1))
            if graph_key in {"block3_exp_n5", "block3_exp_n6"} and cfg_field == "square_rescales" and tuple_index == 3:
                audit_warnings.append(f"{graph_key}: t_new 后续平方 stage 复用 square_rescale_sf_3，这是当前 4 个 square RL 槽的压缩语义")
            if block == 1:
                layer_rule = "layer=1..11；layer 0 的 block1 整体无效且不送 RO"
            elif block == 3:
                layer_rule = f"softmax degree={graph_degree(graph_key)} 的层"
            elif block == 5:
                layer_rule = f"GELU degree={graph_degree(graph_key)} 的层"
            else:
                layer_rule = "layer=0..11"
            t_rows.append({
                "graph_key": graph_key,
                "block": block,
                "t_index": idx,
                "ro_i": base_row.get("i", ""),
                "ro_name": base_row.get("name", ""),
                "ro_type": base_row.get("type", ""),
                "baseline_sf": base_row.get("baseline_sf", ""),
                "baseline_source": base_row.get("source", ""),
                "cfg_field": cfg_field,
                "tuple_index": tuple_index,
                "rl_field": rl_field or "",
                "kind": kind,
                "default_max": default_max,
                "local_offset": offset,
                "global_formula": f"layer*73+{offset}" if offset >= 0 else "",
                "node_name": node_map.get(block, {}).get(rl_field or "", ""),
                "note": note,
                "layer_rule": layer_rule,
            })
    return fields, t_rows, audit_errors, audit_warnings


def metric_cards(items: Sequence[Tuple[str, Any, str]]) -> str:
    return '<div class="grid">' + "".join(
        f'<div class="metric"><strong>{esc(value)}</strong><div>{esc(label)}</div><p class="small">{esc(note)}</p></div>'
        for label, value, note in items
    ) + '</div>'


def report_t_new(t_rows: List[dict], audit_errors: List[str], audit_warnings: List[str]) -> str:
    by_graph: Dict[str, List[dict]] = defaultdict(list)
    for row in t_rows:
        by_graph[row["graph_key"]].append(row)
    metrics = metric_cards([
        ("graph 数", len(by_graph), "DEFAULT_CFG_TO_T_NEW_MAP 中登记的 RO graph"),
        ("t_new stage 数", len(t_rows), "所有 graph 的 t_new 条目总数"),
        ("fatal mismatch", len(audit_errors), "长度或字段不存在会算作严重不对齐"),
        ("semantic warning", len(set(audit_warnings)), "当前可解释的压缩/复用语义"),
    ])
    status = '<div class="card ok"><h2>审计结论</h2><p>未发现 fatal 级 t_new 语义错位：每个 t_new 条目都能找到当前 RL 字段，且 static_skeletons 的 baseline stage 数与 bridge 映射长度一致。</p></div>'
    if audit_errors:
        status = '<div class="card risk"><h2>审计结论</h2><p>发现需要修复的 fatal mismatch：</p><ul>' + "".join(f"<li>{esc(x)}</li>" for x in audit_errors) + "</ul></div>"
    if audit_warnings:
        status += '<div class="card warn"><h2>语义提醒</h2><ul>' + "".join(f"<li>{esc(x)}</li>" for x in sorted(set(audit_warnings))) + "</ul></div>"
    sections = [metrics, status]
    for graph_key, rows in by_graph.items():
        block = rows[0]["block"]
        headers = ["t_new 下标", "RO skeleton i", "RO 位置语义", "baseline SF", "bridge cfg 字段", "对应 RL 字段", "适用层"]
        table_rows = []
        for row in rows:
            source_text = "source 输出 scale" if row["baseline_source"] == "sf" and row["ro_i"] == 0 else (
                "该 cut point 后 post-rescale 目标 scale" if row["baseline_source"] == "sf_post" else "baseline scale"
            )
            table_rows.append([
                esc(f't_new[{row["t_index"]}]'),
                esc(row["ro_i"]),
                f'<code>{esc(row["ro_name"])}</code><br><span class="small">{esc(row["ro_type"])}；{esc(source_text)}</span>',
                esc(row["baseline_sf"]),
                f'<code>{esc(row["cfg_field"])}</code>' + (f'[{esc(row["tuple_index"])}]' if row["tuple_index"] is not None else ""),
                f'<code>{esc(row["rl_field"])}</code><br><span class="small">local offset {esc(row["local_offset"])}；{esc(row["note"])}</span>',
                esc(row["layer_rule"]),
            ])
        sections.append(f"<h2>{esc(graph_key)}：{esc(BLOCK_CN[block])}</h2>" + table(headers, table_rows))
    return page(
        "Rescale_Optimizer 各 graph 的 t_new 语义",
        "这个文件只站在 Rescale_Optimizer 一侧说明：每个 graph 的 t_new[r] 到底对应哪一个 skeleton stage，以及 baseline archive 中该 stage 的实际 scaling factor。",
        "\n".join(sections),
    )


def report_actions(fields, node_map) -> str:
    kind_counts = Counter(kind for _offset, _block, _field, kind, _default in fields)
    block_counts = Counter(block for _offset, block, _field, _kind, _default in fields)
    total_per_layer = len(fields)
    raw_total = total_per_layer * NUM_LAYERS + 1
    default_effective = raw_total - 9 - 1
    metrics = metric_cards([
        ("单层动作维度", total_per_layer, "5 个 block 的固定 slot 数"),
        ("bert-base 总动作维度", raw_total, "12 层 × 73 + deprecated first_input"),
        ("默认有效 slot", default_effective, "默认 degree=4/4；扣除 layer0 block1 与 first_input"),
        ("rescale slot/层", kind_counts["R"], "其中 index=0 表示该 BLB rescale 噪声关闭"),
    ])
    formula = """
    <div class="formula">
      <math display="block"><mrow><mtext>global_index</mtext><mo>=</mo><mtext>layer</mtext><mo>×</mo><mn>73</mn><mo>+</mo><mtext>local_offset</mtext></mrow></math>
      <p class="small">first_input 位于 index 876，当前已废弃且始终 ineffective。</p>
    </div>
    """
    summary_rows = []
    for block in range(1, 6):
        b_fields = [r for r in fields if r[1] == block]
        c = Counter(r[3] for r in b_fields)
        summary_rows.append([
            esc(f"Block {block}"),
            esc(BLOCK_CN[block]),
            esc(len(b_fields)),
            esc(", ".join(f"{k}={c[k]}" for k in ("F", "W", "M", "S", "R", "K") if c[k])),
        ])
    action_rows = []
    for offset, block, field, kind, default_max in fields:
        node_name = node_map.get(block, {}).get(field, "")
        if block == 1:
            effective = "layer 0 无效；layer 1..11 有效"
        elif block == 3 and field.startswith("square_rescale_sf_"):
            slot = int(field.rsplit("_", 1)[-1])
            effective = f"softmax degree > {slot} 时有效"
        elif block == 5 and field.startswith("gelu_power_rescale_sf_"):
            slot = int(field.rsplit("_", 1)[-1])
            effective = f"GELU degree > {slot + 1} 时有效"
        elif block == 5 and field.startswith("gelu_coeff_mul_rescale_sf_"):
            slot = int(field.rsplit("_", 1)[-1])
            effective = f"GELU degree > {slot} 时有效"
        else:
            effective = "有效"
        action_rows.append([
            esc(offset),
            esc(f"{offset}, {73 + offset}, ..."),
            esc(f"Block {block}"),
            f"<code>{esc(field)}</code>",
            esc(KIND_CN[kind]),
            esc(LEVELS_BY_KIND[kind]),
            esc(default_max if kind != "K" else "-"),
            f"<code>{esc(node_name)}</code>" if node_name else "",
            esc(level_values(kind, default_max)),
            esc(effective),
        ])
    body = "\n".join([
        metrics,
        formula,
        "<h2>按 block 统计</h2>",
        table(["Block", "语义范围", "slot/层", "kind 统计"], summary_rows),
        "<h2>单层 73 个动作位置全集</h2>",
        '<p class="small">表中 local offset 会按每层重复；bert-base 的 layer 取 0..11。</p>',
        table(["local offset", "示例 global index", "Block", "RL 字段", "动作类型", "挡位数", "默认 max_sf", "代码节点名", "默认挡位含义", "有效性"], action_rows),
        "<h2>first_input 特殊位</h2>",
        table(["global index", "字段", "状态", "说明"], [[876, "<code>first_input_sf</code>", "deprecated / ineffective", "首个 HE 配置视为无损；当前不再安装 first input fresh 噪声，也不送 RO"]]),
    ])
    return page(
        "BLB Stage-2 RL 动作空间全集与位置",
        "这个文件展示当前 RL 动作向量的所有 slot：每层 73 个、bert-base 共 877 维，并说明每个 slot 的 block 位置、动作类型、挡位含义与有效性。",
        body,
    )


def report_mapping(t_rows: List[dict], audit_errors: List[str], audit_warnings: List[str]) -> str:
    status = '<div class="card ok"><h2>对齐检查</h2><p>当前 RL 动作字段与 Rescale_Optimizer 的 t_new stage 可以逐项对齐；未发现需要修改代码的 fatal 错位。</p></div>'
    if audit_errors:
        status = '<div class="card risk"><h2>对齐检查</h2><ul>' + "".join(f"<li>{esc(x)}</li>" for x in audit_errors) + "</ul></div>"
    if audit_warnings:
        status += '<div class="card warn"><h2>保留语义提醒</h2><ul>' + "".join(f"<li>{esc(x)}</li>" for x in sorted(set(audit_warnings))) + "</ul></div>"
    rows = []
    for row in t_rows:
        if row["kind"] == "R":
            runtime = "action=0 时：该 cfg 字段为 None，bridge 用 baseline t_new 回填；action>0 时：使用 RL 解码出的 scaling factor"
        else:
            runtime = "使用 RL 解码出的 scaling factor"
        rows.append([
            esc(row["graph_key"]),
            esc(row["layer_rule"]),
            esc(f't_new[{row["t_index"]}]'),
            f'<code>{esc(row["ro_name"])}</code><br><span class="small">i={esc(row["ro_i"])}；baseline={esc(row["baseline_sf"])}</span>',
            f'<code>{esc(row["rl_field"])}</code><br><span class="small">{esc(BLOCK_CN[row["block"]])}</span>',
            esc(row["global_formula"]),
            esc(KIND_CN.get(row["kind"], row["kind"])),
            esc(runtime),
        ])
    formula = """
    <div class="formula">
      <math display="block"><mrow><msub><mi>a</mi><mi>j</mi></msub><mo>→</mo><msub><mi>sf</mi><mi>j</mi></msub><mo>→</mo><msub><mi>cfg</mi><mrow><mi>b</mi><mo>,</mo><mi>l</mi></mrow></msub><mo>→</mo><msub><mtext>t_new</mtext><mi>r</mi></msub></mrow></math>
      <p class="small">RL action index 先解码为真实 scaling factor，再写入 BlockNoiseConfig；bridge 从 cfg 里按 _SkelEntry 抽取对应值，组成 Rescale_Optimizer 的 t_new。</p>
    </div>
    """
    body = "\n".join([
        status,
        formula,
        "<h2>RL action 到 t_new 的逐项映射</h2>",
        table(["RO graph", "适用层", "t_new", "RO 真实位置", "RL 动作实际位置", "global index 公式", "动作类型", "运行时 scaling factor 来源"], rows),
        '<div class="card warn"><h2>关于 rescale action=0</h2><p>RL 的 rescale index=0 表示“不安装该 BLB rescale 噪声点”，但 Rescale_Optimizer 仍按固定 baseline skeleton 计算模数链。因此当 t_new 绑定的 cfg 字段为 None 时，bridge 会用 baseline t_new 回填，而不是删除 skeleton stage。这与当前“固定 HE 操作，只让 RL 选择噪声/scale”的语义一致。</p></div>',
    ])
    return page(
        "RL 动作到 Rescale_Optimizer t_new 的映射关系",
        "这个文件把两边坐标系统接起来：展示每个 t_new stage 由哪个 RL 动作字段提供 scaling factor、该动作在向量中的位置，以及运行时如何处理 rescale 关闭。",
        body,
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    action_tree = parse_python(ACTION_SPACE)
    bridge_tree = parse_python(BRIDGE)
    block_specs = parse_block_specs(action_tree)
    node_map = parse_node_map(action_tree)
    t_map = parse_t_new_map(bridge_tree)
    static = static_entries()
    fields, t_rows, audit_errors, audit_warnings = build_model(block_specs, node_map, t_map, static)

    outputs = {
        "blb_rescale_optimizer_t_new_semantics.html": report_t_new(t_rows, audit_errors, audit_warnings),
        "blb_rl_action_space_positions.html": report_actions(fields, node_map),
        "blb_rl_to_t_new_mapping.html": report_mapping(t_rows, audit_errors, audit_warnings),
    }
    for name, content in outputs.items():
        (REPORT_DIR / name).write_text(content, encoding="utf-8", newline="\n")

    summary = {
        "outputs": sorted(outputs),
        "t_new_rows": len(t_rows),
        "audit_errors": audit_errors,
        "audit_warnings": sorted(set(audit_warnings)),
    }
    (REPORT_DIR / "blb_mapping_report_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if audit_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
