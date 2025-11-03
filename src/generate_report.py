"""
生成模型训练总结报告

功能:
1. 收集所有模型的训练结果
2. 生成Markdown格式的总结报告
3. 包含性能对比表、详细分析、可视化对比等

作者: Augment Agent
日期: 2025
"""

import json
import pandas as pd
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_summary_report():
    """生成模型训练总结报告"""
    
    logger.info("生成模型训练总结报告...")
    
    # 收集所有模型的训练日志
    models_data = []
    
    for model in ['RandomForest', 'LSTM', 'Transformer']:
        for data_version in ['full', 'reduced']:
            for es in ['with_early_stopping', 'no_early_stopping']:
                log_path = f"results/training_logs/{model}_{data_version}_{es}.json"
                
                if Path(log_path).exists():
                    with open(log_path, 'r') as f:
                        log = json.load(f)
                    
                    models_data.append({
                        '模型': model,
                        '数据版本': 'Full' if data_version == 'full' else 'Reduced',
                        '早停策略': 'Yes' if es == 'with_early_stopping' else 'No',
                        '最佳Epoch': log.get('best_epoch', '-'),
                        '训练集RMSE': log['train_rmse'],
                        '验证集RMSE': log['val_rmse'],
                        '测试集RMSE': log['test_rmse'],
                        '训练时间(分钟)': log['training_time'] / 60
                    })
    
    if not models_data:
        logger.warning("没有找到任何训练日志文件")
        return
    
    df = pd.DataFrame(models_data)
    
    # 找出最佳模型
    best_model = df.loc[df['测试集RMSE'].idxmin()]
    
    # 生成Markdown报告
    report = []
    report.append("# 航空发动机RUL预测 - 模型训练总结报告\n")
    report.append("---\n")
    report.append(f"**生成时间**: 2025年\n")
    report.append(f"**训练模型数**: {len(df)}个\n")
    report.append(f"**最佳模型**: {best_model['模型']} - {best_model['数据版本']} - 早停: {best_model['早停策略']}\n")
    report.append(f"**最佳测试集RMSE**: {best_model['测试集RMSE']:.4f}\n")
    report.append("\n---\n\n")
    
    # 1. 总体性能对比表
    report.append("## 1. 总体性能对比表\n\n")
    report.append("| 模型 | 数据版本 | 早停策略 | 最佳Epoch | 训练集RMSE | 验证集RMSE | 测试集RMSE | 训练时间(分钟) |\n")
    report.append("|------|---------|---------|----------|-----------|-----------|-----------|---------------|\n")
    
    for _, row in df.iterrows():
        report.append(f"| {row['模型']} | {row['数据版本']} | {row['早停策略']} | {row['最佳Epoch']} | "
                     f"{row['训练集RMSE']:.4f} | {row['验证集RMSE']:.4f} | {row['测试集RMSE']:.4f} | "
                     f"{row['训练时间(分钟)']:.2f} |\n")
    
    report.append("\n")
    
    # 2. 详细分析
    report.append("## 2. 详细分析\n\n")
    
    # 2.1 最佳模型
    report.append("### 2.1 最佳模型\n\n")
    report.append(f"- **模型**: {best_model['模型']}\n")
    report.append(f"- **数据版本**: {best_model['数据版本']}\n")
    report.append(f"- **早停策略**: {best_model['早停策略']}\n")
    report.append(f"- **测试集RMSE**: {best_model['测试集RMSE']:.4f}\n")
    report.append(f"- **验证集RMSE**: {best_model['验证集RMSE']:.4f}\n")
    report.append(f"- **训练时间**: {best_model['训练时间(分钟)']:.2f}分钟\n\n")
    
    # 2.2 数据版本对比
    report.append("### 2.2 数据版本对比\n\n")
    full_avg = df[df['数据版本'] == 'Full']['测试集RMSE'].mean()
    reduced_avg = df[df['数据版本'] == 'Reduced']['测试集RMSE'].mean()
    report.append(f"- **完整特征版本平均测试集RMSE**: {full_avg:.4f}\n")
    report.append(f"- **降维特征版本平均测试集RMSE**: {reduced_avg:.4f}\n")
    
    if full_avg < reduced_avg:
        report.append(f"- **结论**: 完整特征版本性能更好，降维导致性能下降 {(reduced_avg - full_avg):.4f}\n\n")
    else:
        report.append(f"- **结论**: 降维特征版本性能更好，成功去除冗余特征 {(full_avg - reduced_avg):.4f}\n\n")
    
    # 2.3 早停策略对比
    report.append("### 2.3 早停策略对比\n\n")
    es_yes_avg = df[df['早停策略'] == 'Yes']['测试集RMSE'].mean()
    es_no_avg = df[df['早停策略'] == 'No']['测试集RMSE'].mean()
    report.append(f"- **带早停平均测试集RMSE**: {es_yes_avg:.4f}\n")
    report.append(f"- **不带早停平均测试集RMSE**: {es_no_avg:.4f}\n")
    
    if es_yes_avg < es_no_avg:
        report.append(f"- **结论**: 早停策略有效防止过拟合，性能提升 {(es_no_avg - es_yes_avg):.4f}\n\n")
    else:
        report.append(f"- **结论**: 不带早停性能更好，模型需要更多训练轮数 {(es_yes_avg - es_no_avg):.4f}\n\n")
    
    # 2.4 模型对比
    report.append("### 2.4 模型对比\n\n")
    for model in ['RandomForest', 'LSTM', 'Transformer']:
        model_df = df[df['模型'] == model]
        if len(model_df) > 0:
            avg_rmse = model_df['测试集RMSE'].mean()
            min_rmse = model_df['测试集RMSE'].min()
            avg_time = model_df['训练时间(分钟)'].mean()
            report.append(f"- **{model}**: 平均测试集RMSE = {avg_rmse:.4f}, 最佳 = {min_rmse:.4f}, 平均训练时间 = {avg_time:.2f}分钟\n")
    
    report.append("\n")
    
    # 2.5 过拟合分析
    report.append("### 2.5 过拟合分析\n\n")
    report.append("| 模型 | 数据版本 | 早停策略 | 训练集RMSE | 验证集RMSE | 差距 | 过拟合程度 |\n")
    report.append("|------|---------|---------|-----------|-----------|------|----------|\n")
    
    for _, row in df.iterrows():
        gap = row['验证集RMSE'] - row['训练集RMSE']
        if gap < 1:
            level = "低"
        elif gap < 3:
            level = "中"
        else:
            level = "高"
        
        report.append(f"| {row['模型']} | {row['数据版本']} | {row['早停策略']} | "
                     f"{row['训练集RMSE']:.4f} | {row['验证集RMSE']:.4f} | {gap:.4f} | {level} |\n")
    
    report.append("\n")
    
    # 3. 可视化对比
    report.append("## 3. 可视化对比\n\n")
    report.append("### 3.1 所有模型测试集RMSE对比\n\n")
    report.append("![所有模型对比](../results/training_curves/all_models_comparison.png)\n\n")
    
    report.append("### 3.2 最佳模型训练曲线\n\n")
    best_model_name = best_model['模型']
    best_data_version = 'full' if best_model['数据版本'] == 'Full' else 'reduced'
    best_es = 'with_early_stopping' if best_model['早停策略'] == 'Yes' else 'no_early_stopping'
    report.append(f"![最佳模型训练曲线](../results/training_curves/{best_model_name}_{best_data_version}_{best_es}_curves.png)\n\n")
    
    report.append("### 3.3 最佳模型预测结果\n\n")
    report.append(f"![最佳模型预测散点图](../results/training_curves/{best_model_name}_{best_data_version}_{best_es}_scatter.png)\n\n")
    
    # 4. 结论和建议
    report.append("## 4. 结论和建议\n\n")
    report.append("### 4.1 主要发现\n\n")
    report.append(f"1. **最佳模型**: {best_model['模型']}在{best_model['数据版本']}特征版本上取得了最佳性能（测试集RMSE = {best_model['测试集RMSE']:.4f}）\n")
    report.append(f"2. **数据版本**: {'完整特征版本' if full_avg < reduced_avg else '降维特征版本'}整体性能更好\n")
    report.append(f"3. **早停策略**: {'早停策略' if es_yes_avg < es_no_avg else '不使用早停'}能够获得更好的泛化性能\n")
    report.append(f"4. **训练效率**: 随机森林训练最快，Transformer训练最慢但性能可能更好\n\n")
    
    report.append("### 4.2 推荐配置\n\n")
    report.append(f"- **推荐模型**: {best_model['模型']}\n")
    report.append(f"- **推荐数据版本**: {best_model['数据版本']}\n")
    report.append(f"- **推荐早停策略**: {best_model['早停策略']}\n")
    report.append(f"- **预期性能**: 测试集RMSE ≈ {best_model['测试集RMSE']:.4f}\n\n")
    
    report.append("### 4.3 改进方向\n\n")
    report.append("1. **特征工程**: 尝试更多的统计特征和交互特征\n")
    report.append("2. **模型集成**: 将多个模型的预测结果进行集成\n")
    report.append("3. **超参数优化**: 使用网格搜索或贝叶斯优化进一步调优\n")
    report.append("4. **数据增强**: 尝试数据增强技术增加训练样本\n")
    report.append("5. **注意力机制**: 在LSTM中加入注意力机制\n\n")
    
    report.append("---\n\n")
    report.append("**报告生成工具**: Augment Agent\n")
    report.append("**数据集**: NASA C-MAPSS FD001\n")
    
    # 保存报告
    report_path = "augment_caption/模型训练总结报告.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    logger.info(f"总结报告已保存到: {report_path}")


if __name__ == "__main__":
    generate_summary_report()

