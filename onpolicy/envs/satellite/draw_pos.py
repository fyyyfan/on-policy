import matplotlib
matplotlib.use('Agg') # 使用非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_satellite_and_ground_station(satellite_positions, ground_station_positions, output_file='positions.png'):
    """
    绘制3维立体坐标系，标记地面站坐标和卫星坐标。

    Args:
        satellite_positions (list of tuple): 卫星坐标列表，每个元素为 (x, y, z)。
        ground_station_positions (list of tuple): 地面站坐标列表，每个元素为 (x, y, z)。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制卫星坐标
    satellite_x = [pos[0] for pos in satellite_positions]
    satellite_y = [pos[1] for pos in satellite_positions]
    satellite_z = [pos[2] for pos in satellite_positions]
    ax.scatter(satellite_x, satellite_y, satellite_z, c='blue', label='Satellites', marker='o')

    # 绘制地面站坐标
    ground_x = [pos[0] for pos in ground_station_positions]
    ground_y = [pos[1] for pos in ground_station_positions]
    ground_z = [pos[2] for pos in ground_station_positions]
    ax.scatter(ground_x, ground_y, ground_z, c='red', label='Ground Stations', marker='^')

    # 设置坐标轴标签
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # 添加图例
    ax.legend()

    # 保存图像到文件
    plt.savefig(output_file)
    print(f"图像已保存到 {output_file}")

if __name__ == "__main__":
    # 示例数据
    satellite_positions = [(-3708.153504327113, 3950.0724217496027, -4519.148429311474), 
                       (-6191.872167016669, 2452.239832776905, -2304.8338160290273)]
    ground_station_positions = [(-2440.3851012885766, 4226.870985465937, 4095.4524768435485), 
                            (-2841.87100996883, 4729.667613377867, 3185.6964999999996)]

    visualize_satellite_and_ground_station(satellite_positions, ground_station_positions)