# 六相机全量生产最终报告（2026-09-14）

## 结论

严格同步的 nuScenes 六相机数据已经完成生产、整组隔离、staging 审计、
原子发布、正式路径独立复验和清理后复验。正式目录为：

`/mnt/ojc/workplace2/dataset/nusc_pair_6cam`

最终状态为 `PASS`：38,445 个完整 group，692,010 张逻辑 PNG，所有 group
均恰好包含六个固定相机和 `gt/input/target` 三个角色。train/val 没有 scene
重叠，正式审计候选为 0。

## 最终规模

| 项目 | 数量 |
|---|---:|
| train group | 32,591 |
| val group | 5,854 |
| group 合计 | 38,445 |
| camera view | 230,670 |
| 逻辑 PNG | 692,010 |
| visible 编辑 view | 39,535 |
| invisible `input=target` no-op view | 191,135 |
| 多资产 group | 11,582 |
| group 内资产出现次数 | 54,156 |
| 唯一源文件 inode | 152,194 |

类别统计按“资产在已发布 group 中出现一次”计数；括号内为唯一
`global_uid` 数量：

| 类别 | 出现次数 | 唯一资产 |
|---|---:|---:|
| car | 33,605 | 2,609 |
| cone | 9,455 | 971 |
| truck | 6,006 | 459 |
| van | 3,135 | 227 |
| bus | 1,638 | 137 |
| barrier | 317 | 16 |

## GPU 生产结果

- train baseline：2,711/2,711 个 context 成功，失败 0；补齐 40,015 个缺失
  camera baseline view。
- val baseline：878/878 个 context 成功，失败 0；补齐 13,189 个缺失
  camera baseline view。
- train visible backfill：406/406 个 render job 正常结束；严格 compose 接受
  124，拒绝 282。
- val visible backfill：50/50 个 render job 正常结束；严格 compose 接受 26，
  拒绝 24。
- 主生产使用 GPU0-3。GPU4-7 空闲后并行预取 val visible shards；主
  supervisor 随后验证并复用 4/4 个已签名分片，没有重复渲染。
- 三个冻结 checkpoint 的 SHA-256 为：
  - STORM：`141d66c31b0e74a2d7674a37882826ee99168afc8037dd0a36482f5827ba3fbd`
  - CVAC：`256cf70203af71609fe2fd2fa854a243311aaacf11f74e75192b1f552c9f3ca8`
  - DCN：`eaedc491eda4200ebc728831a211980696b07a0b324ba6290abdd92b7005d08b`

渲染成功只表示任务按合同结束；compose 的身份、几何、接地、外观、背景一致性
和遮挡门禁仍会拒绝不合格 view。没有为凑数量重试或替换少数失败样本。

## 整组隔离

52,009 个候选 group 中，13,564 个被整组隔离，没有任何残缺组进入正式目录：

| 首个阻断原因 | group 数 |
|---|---:|
| baseline backfill 不可用或质量门禁失败 | 12,985 |
| visible backfill 被拒绝或不可用 | 409 |
| visible backfill 与绑定背景不一致 | 136 |
| 已有 triplet 的完整 production lineage 不成立 | 34 |

最后一项来自旧 train 数据中的 `metadata_hash_mismatch` 或
`release_authority_only` 条目。它们缺少合格 production record/metadata；新门禁
要求 `lineage_status=complete`、`quality_gate_pass=true` 且 production lineage
齐全。34 个候选被该门禁阻断；其中 8 个在先前版本中本来已因其他原因隔离，
因此最终已发布 group 净减少 26。

## 溯源与审计

正式 metadata 包含：

- `metadata/train_groups.jsonl` 与 `metadata/val_groups.jsonl`；
- `metadata/source_index/train` 与 `metadata/source_index/val` 的完整严格索引；
- `metadata/receipts` 中的渲染结果、配置、production state 和内容哈希清单；
- `metadata/audit` 中的 staging 审计、候选和 quarantine；
- `metadata/reports` 中的阶段 A 报告和发布统计；
- 完整的根目录 `README.md`，含定义、相机顺序、命名、统计、哈希、运行、
  断点续跑和复验命令。

每个正式 group receipt 显式绑定 scene、sample token、frame、context、全局资产
并集、`obj_id`、`instance_token`、`global_uid`、PLY 路径/哈希、checkpoint、
camera exposure、visible/invisible 子集、combination、production record、源角色
路径/哈希、正式相对路径和审查结论。字段覆盖复查遍历 38,445 group、230,670
view 和 692,010 role receipt，错误为 0。

关键审计证据：

- staging 审计：`metadata/audit/summary.json`，`PASS`，candidate 0；
- 独立正式复验：
  `/mnt/ojc/workplace2/sixcam_production_20260913/final_strict_receipt/independent_audit/summary.json`；
- 清理后复验：
  `/mnt/ojc/workplace2/sixcam_production_20260913/final_strict_receipt/post_cleanup_audit/summary.json`；
- 两次正式路径复验摘要 SHA-256 均为
  `822d29420f75df68824c5e5385d4d994e5894ff7dcb5a530e1d18677705f63a7`；
- train group manifest SHA-256：
  `a5b55587550ccd863b0d08db69523a672ce2ecab47d5db977048c9bec61f1734`；
- val group manifest SHA-256：
  `d05bb7c1afb29b5448973a1540f35ff5df94049621c030f400070e4976096e48`；
- quarantine manifest SHA-256：
  `a16b160cb71214d223aaa3286a55589a55d8c725d17d6d2e489a7a93fdade55a`。

## Pilot

真实 pilot 选择 12 个 group，覆盖 train/val、直接复用、补渲染、单/多资产、
多相机可见和 no-op。按严格门禁发布 8 个 group（144 张），4 个失败 group
直接放弃。8 张 6x3 contact sheet 已逐组人工检查；独立审计候选 0。有效 pilot
证据保留在：

`/mnt/ojc/workplace2/sixcam_production_20260913/pilot_lwh`

## 清理

最终审计通过后，先扫描正式 metadata 对候选路径的引用，确认引用数全部为 0，
再删除：

- 两份被最终版本替代的 `.nusc_pair_6cam.previous.*` 发布备份；
- `index_fast`、`index_fast_corrected`、`index_fast_final` 和旧
  `index_verified`；
- width/length 顺序错误的旧 `pilot`；
- 中间 `final_enriched_receipt`；
- 重复的 `val_visible_composed_prefetch` 和预取日志；
- 首次 Python 环境错误产生的四个小 `attempt0000`。

这些路径显示约 34 GiB 逻辑大小；由于 PNG 使用 hardlink，物理可用空间实际
增加约 4 GiB（154 GiB 到 158 GiB）。删除后又完成一次全量正式复验并通过。

保留项包括正式数据、`index_verified_lwh`、有效 pilot、正式图像仍引用的
baseline/compose 源、完整生产断点缓存、原始 production receipt 和最终严格
receipt。原始 `/mnt/ojc/workplace2/dataset/nusc_pair` 以及
`/mnt/ojc/workplace2/DriveHarm` 未修改。

## 复验命令

```bash
/mnt/ojc/miniconda3/envs/nuscenes_localization_v2_splatad/bin/python \
  -m driveharm.cli sixcam-strict-audit \
  --dataset-root /mnt/ojc/workplace2/dataset/nusc_pair_6cam \
  --output-root <new-audit-directory> \
  --workers 32
```

代码、测试、README 和本报告提交到独立仓库；数据、资产、权重、运行缓存和
大日志不进入 Git。
