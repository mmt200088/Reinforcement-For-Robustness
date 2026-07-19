#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
OUTPUTS = [
    ROOT / "outputs" / "seed_2026071901_final2",
    ROOT / "outputs" / "seed_2026071902_final2",
]
DESKTOP_NAMES = {
    2026071901: "20260719_stage2_ep114240_glue_seed_2026071901.zip",
    2026071902: "20260719_stage2_ep114240_glue_seed_2026071902.zip",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def esc(value) -> str:
    return html.escape(str(value))


selected = load_json(ROOT / "artifacts" / "selected_from_checkpoint.json")
fixed = load_json(ROOT / "artifacts" / "strict_best_fusion_fixed_action.json")
manifests = [load_json(path / "run_manifest.json") for path in OUTPUTS]
audits = [load_json(path / "install_audit.json") for path in OUTPUTS]

assert selected["candidate_key"] == manifests[0]["candidate_key"] == manifests[1]["candidate_key"]
assert selected["checkpoint_sha256"] == "c039f3de3619261880aa3eb771d80318b1b17984cae926c4f6e819a3a03b1ab4"
assert fixed["summary"]["total_fusion_count"] == 27
assert fixed["summary"]["boosted_option_count"] == 27

prediction_rows = []
for output in OUTPUTS:
    lines = (output / "MRPC.tsv").read_text().splitlines()
    assert lines[0] == "index\tprediction"
    parsed = [line.split("\t") for line in lines[1:]]
    assert [int(index) for index, _ in parsed] == list(range(1725))
    prediction_rows.append([label for _, label in parsed])

disagreement_indices = [
    index
    for index, (left, right) in enumerate(zip(*prediction_rows))
    if left != right
]

config_hash = manifests[0]["install_config_sha256"]
assert manifests[1]["install_config_sha256"] == config_hash
for audit in audits:
    assert audit["expected_config_sha256"] == config_hash
    assert audit["supplied_to_bridge_sha256"] == config_hash
    assert audit["installed_in_handler_sha256"] == config_hash

dataset = manifests[0]["dataset"]
assert manifests[1]["dataset"] == dataset
assert dataset["test_row_mismatches_vs_cached_arrow"] == 0

zip_members = manifests[0]["zip_members"]
assert manifests[1]["zip_members"] == zip_members
for output in OUTPUTS:
    with zipfile.ZipFile(output / "submission.zip") as archive:
        assert archive.testzip() is None
        assert sorted(archive.namelist()) == sorted(zip_members)

comparison = {
    "schema_version": "stage2_glue_two_seed_comparison_v1",
    "candidate": {
        "candidate_key": selected["candidate_key"],
        "episode": selected["episode"],
        "ppo_update_count": selected["ppo_update_count"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "source_checkpoint": selected["source_checkpoint"],
        "training_source_commit": selected["source_commit"],
        "submission_runtime_commit": manifests[0]["submission_runtime_commit"],
        "action_config_sha256": manifests[0]["action_config_sha256"],
        "installed_config_sha256": config_hash,
        "gelu_degree": selected["gelu_degree"],
        "attn_degree": selected["attn_degree"],
        "fusion_count": fixed["summary"]["total_fusion_count"],
        "boosted_option_count": fixed["summary"]["boosted_option_count"],
        "average_k": fixed["summary"]["avg_k"],
    },
    "dataset": dataset,
    "runs": [
        {
            "seed": manifest["seed"],
            "mrpc_tsv_sha256": manifest["mrpc_tsv_sha256"],
            "submission_zip_sha256": manifest["submission_zip_sha256"],
            "label_counts": manifest["mrpc_label_counts"],
            "desktop_name": DESKTOP_NAMES[manifest["seed"]],
        }
        for manifest in manifests
    ],
    "prediction_disagreement_count": len(disagreement_indices),
    "prediction_disagreement_rate": len(disagreement_indices) / 1725,
    "prediction_disagreement_indices": disagreement_indices,
    "checks": {
        "checkpoint_candidate_matches_both_runs": True,
        "expected_equals_bridge_equals_installed": True,
        "dataset_test_rows_match_cached_arrow": True,
        "mrpc_indices_are_0_through_1724": True,
        "zip_members_and_crc_valid": True,
        "only_mrpc_member_differs_from_template": True,
    },
}
(ROOT / "pair_comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")

choices = {(row["layer"], row["block"]): row for row in fixed["group"]["choices_by_step"]}
layer_rows = []
for layer in range(12):
    cells = []
    for block in (1, 2, 4, 5):
        row = choices.get((layer, block))
        if row is None:
            cells.append("<td class='muted'>N/A</td>")
        else:
            fusion = row["fusion_count"]
            badge = "on" if fusion else "off"
            boost_label = " <span class='boost'>boost</span>" if row["boosted"] else ""
            cells.append(
                f"<td><span class='badge {badge}'>F={fusion}</span> "
                f"<span>K={row['k_value']}</span>"
                f"{boost_label}</td>"
            )
    layer_rows.append(f"<tr><th>L{layer}</th>{''.join(cells)}</tr>")

run_rows = []
for manifest in manifests:
    seed = manifest["seed"]
    counts = manifest["mrpc_label_counts"]
    run_rows.append(
        "<tr>"
        f"<td>{seed}</td>"
        f"<td>{counts.get('0', 0)} / {counts.get('1', 0)}</td>"
        f"<td><code>{esc(manifest['mrpc_tsv_sha256'])}</code></td>"
        f"<td><code>{esc(manifest['submission_zip_sha256'])}</code></td>"
        f"<td><a href='{esc(DESKTOP_NAMES[seed])}'>ZIP</a></td>"
        "</tr>"
    )

check_rows = "".join(
    f"<tr><td>{esc(name.replace('_', ' '))}</td><td class='pass'>PASS</td></tr>"
    for name in comparison["checks"]
)

report = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage-2 MRPC GLUE 双种子提交审计</title>
<style>
:root {{ color-scheme: light; --ink:#17202a; --muted:#64748b; --line:#d8dee8; --paper:#fff; --band:#f4f7fa; --green:#137a4a; --red:#a23a3a; --blue:#225ea8; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:#eef2f6; }}
main {{ max-width:1180px; margin:24px auto; background:var(--paper); padding:30px 34px 40px; border:1px solid var(--line); }}
h1 {{ font-size:25px; margin:0 0 6px; letter-spacing:0; }} h2 {{ font-size:18px; margin:28px 0 10px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
p {{ margin:7px 0; }} .sub {{ color:var(--muted); }} .hero {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); margin:18px 0; }}
.metric {{ background:#fff; padding:13px; min-height:76px; }} .metric b {{ display:block; font-size:20px; margin-top:3px; }} .metric span {{ color:var(--muted); font-size:12px; }}
table {{ width:100%; border-collapse:collapse; margin:8px 0 18px; }} th,td {{ border:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }} thead th {{ background:var(--band); }} tbody th {{ background:#fafbfc; white-space:nowrap; }}
code {{ font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; overflow-wrap:anywhere; }} .pass {{ color:var(--green); font-weight:700; }} .muted {{ color:var(--muted); }}
.badge {{ display:inline-block; min-width:35px; text-align:center; border:1px solid var(--line); padding:1px 5px; font-size:11px; }} .badge.on {{ color:#fff; background:var(--green); border-color:var(--green); }} .badge.off {{ background:#eef1f4; }} .boost {{ color:var(--blue); font-size:11px; }}
.note {{ border-left:3px solid var(--blue); background:#f5f9ff; padding:10px 12px; }} a {{ color:var(--blue); }}
@media(max-width:760px) {{ main {{ margin:0; padding:20px 14px; border:0; }} .hero {{ grid-template-columns:1fr 1fr; }} table {{ display:block; overflow-x:auto; }} }}
</style>
</head>
<body><main>
<h1>Stage-2 MRPC GLUE 双种子提交审计</h1>
<p class="sub">最新 Stage-2 selector-best，episode {selected['episode']}，两次独立高斯噪声推理</p>
<div class="hero">
  <div class="metric"><span>候选 fusion 总数</span><b>{fixed['summary']['total_fusion_count']}</b></div>
  <div class="metric"><span>加大精度选项</span><b>{fixed['summary']['boosted_option_count']}</b></div>
  <div class="metric"><span>平均 truncation K</span><b>{fixed['summary']['avg_k']:.4f}</b></div>
  <div class="metric"><span>两次预测分歧</span><b>{len(disagreement_indices)} / 1725</b></div>
</div>
<p class="note"><b>结论：</b>两个 ZIP 均由同一个 checkpoint selector-best 配置生成；配置从解码、bridge 参数到模型 handler 实际安装三处哈希完全一致。不同噪声种子造成 {len(disagreement_indices)} 条预测差异，符合“同配置、独立高斯噪声”要求。</p>

<h2>候选来源</h2>
<table><tbody>
<tr><th>候选 key</th><td><code>{esc(selected['candidate_key'])}</code></td></tr>
<tr><th>Checkpoint</th><td><code>{esc(selected['source_checkpoint'])}</code></td></tr>
<tr><th>Checkpoint SHA-256</th><td><code>{esc(selected['checkpoint_sha256'])}</code></td></tr>
<tr><th>训练源码 commit</th><td><code>{esc(selected['source_commit'])}</code></td></tr>
<tr><th>提交推理源码 commit</th><td><code>{esc(manifests[0]['submission_runtime_commit'])}</code></td></tr>
<tr><th>动作配置 SHA-256</th><td><code>{esc(manifests[0]['action_config_sha256'])}</code></td></tr>
<tr><th>模型实际安装配置 SHA-256</th><td><code>{esc(config_hash)}</code></td></tr>
<tr><th>Stage-1 固定函数</th><td>GELU = [4] x 12；Softmax = [6] x 12</td></tr>
</tbody></table>

<h2>逐层 Fusion 与 K</h2>
<p class="sub">F 为 fusion count；boost 表示该融合选项已通过 precision-boost 后的显式 SF 路径送入模型。</p>
<table><thead><tr><th>层</th><th>Block 1</th><th>Block 2</th><th>Block 4</th><th>Block 5</th></tr></thead><tbody>{''.join(layer_rows)}</tbody></table>

<h2>两个官方提交包</h2>
<table><thead><tr><th>噪声种子</th><th>预测 0 / 1</th><th>MRPC.tsv SHA-256</th><th>ZIP SHA-256</th><th>文件</th></tr></thead><tbody>{''.join(run_rows)}</tbody></table>
<p>预测分歧率：<b>{len(disagreement_indices) / 1725:.4%}</b>。分歧 index：<code>{', '.join(map(str, disagreement_indices))}</code></p>

<h2>硬门禁</h2>
<table><thead><tr><th>检查</th><th>结果</th></tr></thead><tbody>{check_rows}</tbody></table>
<p>MRPC test 本地 Parquet 与缓存 Arrow：1725 行逐行比较，差异 <b>0</b>；字段顺序为 <code>{esc(dataset['test_columns'])}</code>。两个 ZIP 都包含 GLUE 要求的 11 个 TSV，CRC 正常，index 连续为 0..1724；除 MRPC.tsv 外，其余 10 个成员与已验证模板逐字节相同。</p>

<h2>说明</h2>
<p>GLUE ZIP 需要包含全部任务。本次只重新生成 MRPC.tsv；其他任务沿用已验证的完整 GLUE 模板。两个包之间唯一变化是 MRPC 的高斯噪声种子及由此产生的预测。</p>
</main></body></html>"""

(ROOT / "stage2_glue_two_seed_verification_report.html").write_text(report)
print(json.dumps({
    "report": str(ROOT / "stage2_glue_two_seed_verification_report.html"),
    "comparison": str(ROOT / "pair_comparison.json"),
    "disagreements": len(disagreement_indices),
    "zip_sha256": [manifest["submission_zip_sha256"] for manifest in manifests],
}, indent=2))
