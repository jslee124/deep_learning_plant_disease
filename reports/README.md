# 报告与提交材料

```text
reports/
├── experiment_report/       # 平时实验报告：封面 + 实验一 CNN + 实验二 RNN
├── course_design_report/    # 期末课程设计报告
└── source_code/             # 平时实验独立源码提交目录
```

当前平时实验报告和课程设计报告均保存为 Markdown 正文，并将生成 Word/PDF 正文版。
学校统一封面、姓名、学号、班级和教师信息待模板下发后添加。源代码提交包不包含数据、
模型和输出，以控制体积。

最终正文文件：

- `experiment_report/deep_learning_experiment_report_body.docx`
- `experiment_report/deep_learning_experiment_report_body.pdf`
- `course_design_report/course_design_report_body.docx`
- `course_design_report/course_design_report_body.pdf`

重新生成 Word：

```powershell
.\venv\Scripts\python.exe reports\build_word_reports.py
```
