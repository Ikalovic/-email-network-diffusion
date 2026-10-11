# 组织邮件网络中传播起点选择与跨部门覆盖的仿真研究

本仓库提供社会计算课程论文的实验代码、公开数据与仿真结果，比较全局高出度、部门分散与均匀随机三种起点选择方法。实验在 email-Eu-core 组织邮件网络上采用独立级联模型，考察新增接收人数与跨部门覆盖。

## 已完成的实验

- 数据：email-Eu-core，1,005 个节点，42 个部门。
- 清洗：25,571 条原始有向边，删除 642 条自环，得到 24,929 条有向边；保留所有标签节点，包括 19 个孤立节点。
- 配置：预算 k=5、10；传播概率 p=0.01、0.05、0.10；每个策略与参数组合 500 次。
- 总计：18 个配置，9,000 次终态计算；随机数主种子 20261009。
- 正确性：小图精确枚举与独立逐轮 IC 实现对照通过，脚本见 validation/。

## 运行

克隆仓库并进入目录：

```bash
git clone https://github.com/Ikalovic/-email-network-diffusion.git email-network-diffusion
cd email-network-diffusion
```

建议使用 Python 3.12。固定依赖版本见 requirements.txt。安装依赖后，主实验使用包内数据，可离线运行。在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python run_experiment.py
python plot_results.py
python validation/audit_diffusion_independent.py
```

主实验默认覆盖 results/ 中的同名结果。若要保留已交付结果并单独复跑：

```bash
python run_experiment.py --results-dir results_reproduced
```

plot_results.py 默认读取 results/。图中的英文标签对应：Global degree=全局高出度，Dept. dispersed=部门分散，Random=均匀随机。

固定数据、依赖与种子可以复现抽样结果；运行耗时、平台和完成时间会改变。保存结果对应 2026 年 10 月 9 日的实验记录，环境为 Python 3.12.14，主程序实际用时约 3.63 秒。

## 数据与选种

边 u→v 表示 u 曾给 v 发送过至少一封邮件。实验保留方向、不加权、不截取最大连通分量。

全局高出度按清洗后出度降序选前 k 人，并列时按原始节点 ID 升序。部门分散扫描同一排序，每个部门最多选一人，直到 k 人。随机策略每轮在全部节点中无放回均匀抽样，包含可能抽到孤立节点的情况。

同一轮中，k=5 种子是 k=10 种子的子集。随机策略通过独立 RNG 流生成节点排列，再分别取前 5 和前 10 个节点。

## 传播模型与随机比较

每个新激活节点对每个尚未激活的出邻居仅尝试一次，成功概率为 p。失败后不重试。最终保持激活。

程序使用等价的 live-edge 方法：每条清洗后的有向边生成独立 U(0,1) 随机数，U<p 则该边可通行，从种子做有向可达性搜索。同一 trial 的所有策略、预算和概率共享边随机数；随机选种由另一条独立子流产生。不同 trial 独立抽样。

传播概率是敏感性分析设定，不是从真实转发记录拟合的参数。BFS 波次对应模型轮次，不是实际小时或天。

## 指标字典

记 S 为种子集合、A 为最终激活集合、V_d 为部门 d 全体成员。

| 指标或派生量 | 定义 | 使用方式 |
|---|---|---|
| new_reached | G = \|A\\S\| | 同预算下的主要人数指标，排除种子节点 |
| total_reached（派生） | T = \|A\| = G+k | 由 new_reached 与 k 计算，用于比较预算效果 |
| new_reached_fraction | G/(1005-k) | 在非种子总体中的新增比例 |
| departments_with_new_reach | 至少有一个非种子节点被激活的部门数 | 主要部门指标，称为“产生新增接收者的部门数” |
| macro_department_new_fraction | 42 个部门的新增人数/该部门固定总人数，再取平均 | 补充描述，不直接解释为公平性 |

“排除种子”仅排除节点，不排除其所在的整个部门。种子部门中如果激活了其他成员，该部门仍计入部门指标。只有种子被激活时，两项主指标均为 0。

相同 k 下按 new_reached 比较，与按 total_reached 比较给出相同的人数差。跨 k 必须报告 total_reached，避免额外扣除种子造成表面下降。

## 统计口径

每个配置报告 500 次均值、样本标准差、Monte Carlo 标准误和近似 95% MC 区间（均值 ± 1.96 × 标准误）。策略差先在相同 trial 内相减，再计算差值的标准误，属于配对比较。

这些区间只反映当前固定网络、模型与参数下的模拟均值误差，不是现实组织传播效果的总体置信区间，也没有进行多重检验校正。主要论述关注预先确定的人数和部门指标及实际差异大小。

500 次结果再按顺序分为 5 个 100 次区块，观察主策略差异的方向是否稳定。这是对既有结果的稳定性描述，不是增加五次独立主实验。

## 核心结果

在 p=0.01 时，部门分散相对高出度的平均新增人数差为 -3.376（k=5）与 -4.604（k=10），部门指标也较低。两种预算的五个区块方向一致。

在 p=0.05 时，部门分散人数分别多 0.334 与 1.030，均不足 0.3%；在 p=0.10 时仅多 0.004 与 0.020，部门指标在本次试验中相同。

随机策略在 k=10、p=0.05 或 0.10 时人数均值略高于两个固定策略，优势不到 1%；在 k=5、p=0.05 时，随机策略有 89.8% 的试验新增人数高于全局高出度策略，但其平均新增人数低 27.968 人。

高概率下增加高出度起点的最终总人数收益很小。p=0.10 时 k=5→10 仅增加 0.004 人，而不是触达减少。

## 目录

| 文件 | 内容 |
|---|---|
| run_experiment.py | 数据清洗、选种、传播、统计、配置与耗时 |
| plot_results.py | 从保存结果生成展示表和图，不重跑传播 |
| requirements.txt | 本次使用的依赖版本 |
| data/ | 原始公开边、部门标签、来源说明 |
| results/runs.csv | 9,000 行逐次结果 |
| results/summary.csv | 18 个配置的汇总统计 |
| results/paired_differences.csv | 三组两两策略差值、区间、胜出和相等比例 |
| results/block_stability.csv | 部门分散减全局高出度的五区块差值 |
| results/seed_diagnostics.csv | 四组固定种子的成员及结构特征 |
| results/random_seed_diagnostics.csv | 每轮随机种子的成员及结构特征 |
| results/department_results.csv | 每部门平均新增人数及比例 |
| results/diffusion_curves.csv | 模型波次的累计平均新增人数 |
| results/budget_differences.csv | 增加起点预算的配对最终总人数差异 |
| results/report_table.csv | 论文展示用的主要均值表 |
| results/network_audit.json | 清洗统计与来源 |
| results/experiment_config.json | 参数、随机数、指标口径、版本和耗时 |
| figures/ | 3 张科研图，均有 PNG 和矢量 PDF |
| validation/ | 小图精确核验脚本及实际通过记录 |

## 参考资料

1. Kempe D, Kleinberg J, Tardos É. Maximizing the Spread of Influence through a Social Network. KDD 2003, 137–146. DOI: 10.1145/956750.956769. https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf
2. Weng L, Menczer F, Ahn Y-Y. Virality Prediction and Community Structure in Social Networks. Scientific Reports, 2013, 3:2522. DOI: 10.1038/srep02522. https://arxiv.org/abs/1306.0158
3. Yin H, Benson A R, Leskovec J, Gleich D F. Local Higher-Order Graph Clustering. KDD 2017, 555–564. DOI: 10.1145/3097983.3098069.
4. Leskovec J, Kleinberg J, Faloutsos C. Graph Evolution: Densification and Shrinking Diameters. ACM TKDD, 2007, 1(1), Article 2. DOI: 10.1145/1217299.1217301.
5. Stanford SNAP. email-Eu-core network. https://snap.stanford.edu/data/email-Eu-core.html （2026-10-09访问）
