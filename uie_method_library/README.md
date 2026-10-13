# 水下图像增强统一方法程序库

本程序库把“方法名称、程序来源、权重来源、环境、适配器和运行状态”分开管理。登记进入`registry.csv`不代表已经复现；只有`ready_verified`状态允许进入正式基准。

## 当前规模

- 经典/物理/融合方法：20种，另有RAW对照。
- 深度网络候选：50种。
- 当前可执行并通过统一四域冒烟测试：28个深度网络配置；每个配置均取得4个成功输出，共112个最终成功验证样本。兼容性调试中的历史失败单独保留，不从原始报告中删除。
- 计数口径：UWCNN不同水体权重和UVE-Net大小模型按“网络配置”计数，不作为独立算法重复宣称；28个配置对应20个独立深度算法组。
- 可执行方法总数：RAW对照1种、经典/物理/融合方法20种、深度网络配置28种，共49种，超过“至少30种且至少20种深度网络”的要求。
- 8数据集正式复测：28×217=6076个逐图任务全部成功，形成224个“网络—数据集”汇总单元；测试设备为NVIDIA GeForce GTX 1080 Ti，计时边界见元数据文件。

## CLI

```powershell
# 列出全部深度网络
python -m uie_method_library.run list --deep

# 运行单一方法
python -m uie_method_library.run run `
  --method D001 `
  --input input.jpg `
  --output output.png `
  --metadata run.json

# 28个已就绪网络在8个数据集上断点续跑
python scripts/benchmark_method_library_deep_8datasets.py --resume --save-samples
```

程序不允许把缺失权重的方法自动替换成其他网络。未就绪方法会返回包含状态和阻塞原因的明确错误。

## 状态定义

- `ready_verified`：仓库、权重、依赖、适配器和单图输出均已验证。
- `ready_verified_*`：已通过基础验证，但存在便携预处理、几何修正、单帧或替代语义输入等必须保留的协议限定。
- `adapter_debugging`：权重存在，正在验证输入归一化、输出范围和模型结构。
- `adapter_pending`：仓库和权重存在，但统一适配器尚未完成。
- `weights_missing`：程序存在，官方权重尚未获取。
- `repository_missing`：论文中候选地址尚未验证或当前不可访问。

正式8库测试只读取`ready_verified`方法，并保留逐图失败日志、运行元数据和权重哈希。

## 主要产物

- `registry.csv`：71个登记项的代码、权重、来源、适配器和状态。
- `../benchmark_assets/55_method_library_status_audit.csv`：可执行配置和独立算法组审计。
- `../benchmark_assets/72_method_library_deep28_8datasets_per_image.csv`：6076条逐图指标与运行时间。
- `../benchmark_assets/73_method_library_deep28_8datasets_summary.csv`：224条方法—数据集汇总。
- `../benchmark_assets/74_method_library_deep28_8datasets_failures.csv`：失败日志；本轮仅含表头，表示零失败。
- `../benchmark_assets/75_method_library_deep28_8datasets_metadata.json`：环境、数据规模、指标定义和计时边界。
- `../benchmark_assets/77_method_library_49_8datasets_per_image.csv`：21个经典/物理/对照配置与28个深度配置合并后的10633条逐图记录。
- `../benchmark_assets/78_method_library_49_8datasets_summary.csv`：49种配置在8个数据源上的392条统一汇总。
- `../benchmark_assets/79_method_library_49_8datasets_metadata.json`：合并计数、去重规则、完整性和计时警告。
- `../benchmark_runs/method_library_deep_8datasets/state/`：逐方法断点状态，可安全续跑。
- `../matlab/show_50_method_results.m`：在单张5×10画布中显示输入、参考图、20种经典方法和28个深度网络结果。
- `../matlab/run_show_50_method_results.m`：MATLAB一键启动与300 dpi导出脚本。
