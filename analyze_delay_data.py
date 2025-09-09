#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
延迟数据分析脚本
用于分析训练过程中收集的用户服务延迟数据

使用方法:
python analyze_delay_data.py --data_dir ./results/your_experiment/delay_data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_delay_data(csv_path):
    """加载延迟数据CSV文件"""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"成功加载数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def analyze_delay_trends(df, output_dir):
    """分析延迟趋势"""
    plt.figure(figsize=(15, 10))
    
    # 只分析成功服务的数据
    successful_df = df[df['service_status'] == True].copy()
    
    if len(successful_df) == 0:
        print("没有成功服务的数据可供分析")
        return
    
    # 1. 总延迟随时间变化
    plt.subplot(2, 3, 1)
    episode_delay = successful_df.groupby('episode')['total_delay_ms'].mean()
    plt.plot(episode_delay.index, episode_delay.values)
    plt.title('平均总延迟随Episode变化')
    plt.xlabel('Episode')
    plt.ylabel('平均总延迟 (ms)')
    plt.grid(True)
    
    # 2. 延迟组件分析
    plt.subplot(2, 3, 2)
    delay_components = successful_df[['compute_delay_ms', 'comm_delay_ms', 'migration_delay_ms']].mean()
    plt.bar(delay_components.index, delay_components.values)
    plt.title('平均延迟组件分布')
    plt.ylabel('延迟 (ms)')
    plt.xticks(rotation=45)
    
    # 3. 服务成功率随时间变化
    plt.subplot(2, 3, 3)
    episode_success_rate = df.groupby('episode')['service_status'].mean()
    plt.plot(episode_success_rate.index, episode_success_rate.values)
    plt.title('服务成功率随Episode变化')
    plt.xlabel('Episode')
    plt.ylabel('服务成功率')
    plt.grid(True)
    
    # 4. 用户奖励分布
    plt.subplot(2, 3, 4)
    plt.hist(successful_df['user_reward'], bins=30, alpha=0.7)
    plt.title('用户奖励分布')
    plt.xlabel('用户奖励')
    plt.ylabel('频次')
    
    # 5. 剩余可见时间分布
    plt.subplot(2, 3, 5)
    plt.hist(successful_df['remaining_visibility_time_s'], bins=30, alpha=0.7)
    plt.title('剩余可见时间分布')
    plt.xlabel('剩余可见时间 (s)')
    plt.ylabel('频次')
    
    # 6. 延迟与可见时间的关系
    plt.subplot(2, 3, 6)
    plt.scatter(successful_df['remaining_visibility_time_s'], successful_df['total_delay_ms'], alpha=0.6)
    plt.title('延迟与剩余可见时间关系')
    plt.xlabel('剩余可见时间 (s)')
    plt.ylabel('总延迟 (ms)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delay_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"延迟趋势分析图已保存到: {output_dir}/delay_analysis.png")

def analyze_satellite_performance(df, output_dir):
    """分析各卫星的性能表现"""
    plt.figure(figsize=(12, 8))
    
    successful_df = df[df['service_status'] == True].copy()
    
    if len(successful_df) == 0:
        return
    
    # 按卫星分组分析
    sat_stats = successful_df.groupby('satellite_id').agg({
        'total_delay_ms': ['mean', 'std', 'count'],
        'user_reward': 'mean',
        'service_status': 'count'
    }).round(3)
    
    # 平展列名
    sat_stats.columns = ['_'.join(col) for col in sat_stats.columns]
    sat_stats = sat_stats.reset_index()
    
    # 1. 各卫星平均延迟
    plt.subplot(2, 2, 1)
    plt.bar(sat_stats['satellite_id'], sat_stats['total_delay_ms_mean'])
    plt.title('各卫星平均总延迟')
    plt.xlabel('卫星ID')
    plt.ylabel('平均总延迟 (ms)')
    
    # 2. 各卫星服务次数
    plt.subplot(2, 2, 2)
    plt.bar(sat_stats['satellite_id'], sat_stats['total_delay_ms_count'])
    plt.title('各卫星服务次数')
    plt.xlabel('卫星ID')
    plt.ylabel('服务次数')
    
    # 3. 各卫星平均用户奖励
    plt.subplot(2, 2, 3)
    plt.bar(sat_stats['satellite_id'], sat_stats['user_reward_mean'])
    plt.title('各卫星平均用户奖励')
    plt.xlabel('卫星ID')
    plt.ylabel('平均用户奖励')
    
    # 4. 延迟标准差
    plt.subplot(2, 2, 4)
    plt.bar(sat_stats['satellite_id'], sat_stats['total_delay_ms_std'])
    plt.title('各卫星延迟标准差')
    plt.xlabel('卫星ID')
    plt.ylabel('延迟标准差 (ms)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'satellite_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存统计数据
    sat_stats.to_csv(os.path.join(output_dir, 'satellite_statistics.csv'), index=False)
    print(f"卫星性能分析图已保存到: {output_dir}/satellite_performance.png")
    print(f"卫星统计数据已保存到: {output_dir}/satellite_statistics.csv")

def generate_summary_report(df, output_dir):
    """生成汇总报告"""
    summary = {
        'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_records': len(df),
        'unique_episodes': df['episode'].nunique(),
        'unique_satellites': df['satellite_id'].nunique(),
        'unique_users': df['user_id'].nunique(),
        'overall_statistics': {}
    }
    
    # 总体统计
    successful_df = df[df['service_status'] == True]
    failed_df = df[df['service_status'] == False]
    
    summary['overall_statistics'] = {
        'total_services': len(df),
        'successful_services': len(successful_df),
        'failed_services': len(failed_df),
        'overall_success_rate': len(successful_df) / len(df) if len(df) > 0 else 0,
        'avg_total_delay_ms': successful_df['total_delay_ms'].mean() if len(successful_df) > 0 else 0,
        'std_total_delay_ms': successful_df['total_delay_ms'].std() if len(successful_df) > 0 else 0,
        'avg_compute_delay_ms': successful_df['compute_delay_ms'].mean() if len(successful_df) > 0 else 0,
        'avg_comm_delay_ms': successful_df['comm_delay_ms'].mean() if len(successful_df) > 0 else 0,
        'avg_migration_delay_ms': successful_df['migration_delay_ms'].mean() if len(successful_df) > 0 else 0,
        'avg_user_reward': successful_df['user_reward'].mean() if len(successful_df) > 0 else 0,
        'avg_remaining_visibility_s': successful_df['remaining_visibility_time_s'].mean() if len(successful_df) > 0 else 0
    }
    
    # 按episode的统计
    episode_stats = df.groupby('episode').agg({
        'service_status': ['sum', 'count', 'mean'],
        'total_delay_ms': 'mean',
        'user_reward': 'mean'
    }).round(3)
    
    summary['episode_trends'] = {
        'first_episode': int(df['episode'].min()),
        'last_episode': int(df['episode'].max()),
        'avg_services_per_episode': episode_stats[('service_status', 'count')].mean(),
        'avg_success_rate_trend': episode_stats[('service_status', 'mean')].mean(),
        'delay_improvement': 'improving' if episode_stats[('total_delay_ms', 'mean')].iloc[-1] < episode_stats[('total_delay_ms', 'mean')].iloc[0] else 'worsening'
    }
    
    # 保存报告
    report_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"汇总报告已保存到: {report_path}")
    
    # 打印关键统计信息
    print("\n=== 数据分析汇总 ===")
    print(f"总记录数: {summary['total_records']}")
    print(f"Episode数量: {summary['unique_episodes']}")
    print(f"卫星数量: {summary['unique_satellites']}")
    print(f"用户数量: {summary['unique_users']}")
    print(f"总体服务成功率: {summary['overall_statistics']['overall_success_rate']:.3f}")
    print(f"平均总延迟: {summary['overall_statistics']['avg_total_delay_ms']:.3f} ms")
    print(f"平均用户奖励: {summary['overall_statistics']['avg_user_reward']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='分析延迟数据')
    parser.add_argument('--data_dir', type=str, required=True, help='延迟数据目录路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，默认为数据目录下的analysis文件夹')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'analysis')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找CSV文件
    csv_path = os.path.join(args.data_dir, 'user_delay_data.csv')
    
    if not os.path.exists(csv_path):
        print(f"未找到延迟数据文件: {csv_path}")
        return
    
    # 加载数据
    df = load_delay_data(csv_path)
    if df is None:
        return
    
    print(f"数据概览:")
    print(f"  记录数量: {len(df)}")
    print(f"  时间范围: Episode {df['episode'].min()} - {df['episode'].max()}")
    print(f"  卫星数量: {df['satellite_id'].nunique()}")
    print(f"  用户数量: {df['user_id'].nunique()}")
    print()
    
    # 执行分析
    print("正在分析延迟趋势...")
    analyze_delay_trends(df, args.output_dir)
    
    print("正在分析卫星性能...")
    analyze_satellite_performance(df, args.output_dir)
    
    print("正在生成汇总报告...")
    generate_summary_report(df, args.output_dir)
    
    print(f"\n分析完成！结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 