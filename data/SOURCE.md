# 数据来源

数据集：Stanford Network Analysis Project, email-Eu-core。

- 数据说明：https://snap.stanford.edu/data/email-Eu-core.html
- 有向边：https://snap.stanford.edu/data/email-Eu-core.txt.gz
- 部门标签：https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz
- 获取日期：2026-10-09。

包内两个 txt 文件为公开下载内容的解压版本。清洗在 run_experiment.py 中完成，不修改这两个原始文件。边表示曾经存在的发件关系，标签表示匿名部门。使用数据时应保留对 SNAP 及官方列出的 Yin et al. 2017、Leskovec et al. 2007 的引用，详见 README.md。未给这些数据另行声明新的许可。
