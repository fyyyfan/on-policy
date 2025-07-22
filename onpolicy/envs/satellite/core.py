
import numpy as np
from math import atan2, atan, acos, asin, sin, cos, pi, pow, sqrt, erfc, degrees, radians, log2
from skyfield.api import EarthSatellite, load
from skyfield.toposlib import wgs84
from dataclasses import dataclass
from sympy import symbols, solve

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go

'''
    自定义卫星和用户类
'''
@dataclass
class Time:
    """时间类，用于卫星位置计算"""
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: float = 0.0

    def to_skyfield_time(self):
        """转换为skyfield时间对象"""
        return load.timescale().utc(self.year, self.month, self.day, 
                                  self.hour, self.minute, self.second)

# 服务实例类
class ServiceInstance(object):
    def __init__(self, id, size):
        self.service_id = id
        self.instance_size = size

# 用户簇实体类
class UserCluster(object):
    def __init__(self, id, lon, lat, service_instance: ServiceInstance, task_size):
        # 用户簇索引
        self.id = id
        # 用户地理位置——经度、纬度
        self.lon = lon
        self.lat = lat
        # 用户天线的最大张角
        self.up_ang = int(160)
        # self.val = val # 最小仰角
        # 服务实例
        self.service_instance = service_instance
        # 任务大小
        self.task_size = task_size
        # 当前所在卫星
        self.current_sat = None

    # def is_visible_to(self, sat, min_elevation_angle_deg):
    #     """
    #     TODO: 可见性计算参考卫星平台代码
    #     :param sat:卫星节点对象
    #     计算是否与卫星可见，根据论文公式 (4)(5)
    #     返回可见性起始时间 T_start, 可见时间 T_vis
    #     """
    #     Re = 6371e3  # 地球半径 (m)
    #     h = sat.height
    #     v = sat.velocity  # 卫星速度 (m/s)，粗略值
    #     theta = np.radians(min_elevation_angle_deg)
        
    #     gamma = np.arccos((Re / (Re + h)) * np.cos(theta)) - theta
    #     T_vis = 2 * gamma * (Re + h) / v
    #     return T_vis

class SatelliteNode(object):
    '''卫星实体类'''
    def __init__(self, 
                 id, h, i, comp_resource, tran_power, tran_gain, rec_gain, 
                 tle_line1, tle_line2):
        # 卫星索引
        self.id = id
        # 卫星轨道半径
        self.h = h
        # 卫星轨道倾角
        self.i = i
        # 卫星天线的最大张角
        self.down_ang = int(160) 
        # 卫星计算资源
        self.comp_resource = comp_resource
        # 发射功率
        self.tran_power = tran_power
        # 发射增益
        self.tran_gain = tran_gain
        # 接收增益
        self.rec_gain = rec_gain
        # wgs84库的卫星对象
        self.sat = EarthSatellite(tle_line1, tle_line2)
        # 可见用户簇
        self.visible_user = []
        # 服务的用户族, UserCluster类
        self.service_users = []
        # 部署的服务实例列表,ServiceInstance类
        self.instance_list = []
        # 可见目标卫星列表, Satellite类
        self.target_sat_list = []

    def _satellite_pos(self, time: Time, pos='xyz'):
        """
        计算某时刻卫星对象的位置

        Args:
            yr, mon, day, hr, mins, sec: 年月日时分秒
            pos: 输出格式
                - 'spt': 星下点（经度、纬度、高度）输出
                - 'xyz': WGS84三维坐标（x/y/z）输出

        Return: 
            坐标列表
        """
        t = time.to_skyfield_time()
        geocentric = self.sat.at(t)
        # 转化为wgs84
        wgs84_pos = wgs84.geographic_position_of(geocentric)
        lon = wgs84_pos.longitude.degrees
        lat = wgs84_pos.latitude.degrees
        alt = wgs84_pos.elevation.km
        # 按格式输出
        if pos == 'xyz':
            return [
                (alt + wgs84.radius.km) * cos(lat/180*pi) * cos(lon/180*pi),
                (alt + wgs84.radius.km) * cos(lat/180*pi) * sin(lon/180*pi),
                (alt + wgs84.radius.km) * sin(lat/180*pi)
            ]
            # return geocentric.position.km.tolist()
        return [lon, lat, alt+wgs84.radius.km] 
    
    def _get_visible_user(self, users: UserCluster, time: Time):
        '''
        TODO:
        计算卫星对象的可见用户族
        Args:
            users: 需要进行判断的用户族
            time: 时间
        Return:
            如果可见则返回dist,不可见返回-1
        '''
        pos_sat = self._satellite_pos(time)
        # 获取卫星的可见性阈值
        limit_val = _get_limit_elevation_ang_or_dist(
            self, self.h, users.up_ang, self.down_ang)
        # print("可见性阈值：", limit_val)
        dist = is_visible_or_dist(pos_sat, users.lon, users.lat, limit_val, val_type="elevation_ang")
        if dist > 0:
            return dist
        else:
            return -1


class SatelliteObs(object):
    '''
    卫星智能体观测空间
    '''
    def __init__(self):
        self.task_sizes = None  # W^n_t: List[float], 每个用户簇任务数据大小
        self.neighbor_distances = None # d^n_t: List[float], 与各目标卫星的传输距离
        self.neighbor_data_rates = None # R^n_t: List[float], 与各目标卫星的链路传输速率
        self.neighbor_comp_rsc = None # f^n_t: List[float], 目标卫星可用计算资源
        self.neighbor_visibility_remain_time = None # T^n_t: List[float], 与各目标卫星的剩余可见时间，不确定是否需要

class SatelliteAction(object):
    '''
    卫星智能体动作空间
    '''
    """
    语义友好的动作类：可以表示为具体动作 dict，也可以编码为整数编号，适用于策略网络。
    """
    def __init__(self, action_id: int = 0):
        self.action_id = action_id  # 离散动作编号，0 表示不迁移，默认不迁移

    def decode(self, target_sat_list):
        """
        将离散动作编号解码为语义动作元组 (service_id, target_id)
        Args:
            target_sat_list: 目标卫星列表
        Return: 
            service_id：服务索引
            target_id：目标卫星ID
            (-1, -1) 表示不迁移
        """
        if self.action_id == 0:
            return -1, -1

        index = self.action_id - 1
        target_size = len(target_sat_list)
        service_id = index // target_size
        target_index = index % target_size
        
        # 根据 target_index 获取 target_id
        if target_index < len(target_sat_list):
            target_id = target_sat_list[target_index].id
        else:
            target_id = -1  # 无效的 target_index

        return service_id, target_id

    @staticmethod
    def encode(service_id: int, target_id: int, target_sat_list) -> int:
        """
        将语义动作 (service_id, target_id) 编码为离散动作编号
        Args:
            service_id: 服务索引
            target_id: 目标卫星ID
            target_sat_list: 目标卫星列表
        Returns:
            action_id: 编码后的动作ID
        """
        if service_id < 0 or target_id < 0:
            return 0  # 不迁移
        
        # 根据 target_id 找到对应的 target_index
        target_index = -1
        for i, sat in enumerate(target_sat_list):
            if sat.id == target_id:
                target_index = i
                break
        if target_index == -1:
            return 0  # target_id 不在 target_sat_list 中，返回不迁移
        
        # 使用 target_index 进行编码：1 + service_id * len(target_sat_list) + target_index
        return 1 + service_id * len(target_sat_list) + target_index


# 卫星实体类
class Satellite(SatelliteNode):
    '''
    对卫星属性的封装
    '''
    def __init__(self, id, h, i, comp_resource, tran_power, tran_gain, rec_gain, tle_line1, tle_line2):
        super().__init__(id, h, i, comp_resource, tran_power, tran_gain, rec_gain, tle_line1, tle_line2)
        # 卫星状态空间
        self.state = SatelliteObs()
        # 卫星动作空间
        self.action = SatelliteAction()


### 工具函数
# 卫星基础运行
def _get_limit_elevation_ang_or_dist(self, sat_h, up_ang, down_ang,
                                         output="elevation_ang"):
        """
        获取两设备间的极限值
        值的类型是最小仰角/最大距离
        计算星地链路和不同高度轨道间的星座链路使用

        Args:
            h1, h2: 两设备高度
            up_ang: 位于低处的设备向上看的最大张角
            down_ang: 位于高处的设备向下看的最大张角
            output: 输出内容，"elevation_ang"指输出最小仰角，"dist"指输出最大距离
            
        Returns:
            单位为度的最小仰角
        """
        # 模型准备
        h = sat_h
        R = 6371.393 #地球半径, 表示用户的高度
        K = h * h - R * R
        cos_2 = cos(down_ang / 360 * pi)
        cos_1 = cos(up_ang / 360 * pi)
        
        # 模型求解
        l = symbols('l', real=True)
        f1 = l * l - 2 * l * h * cos_2 + K
        f2 = l * l + 2 * l * R * cos_1 - K
        ans1 = solve([f1])  # 第一个方程的解集，可能0~2个解
        ans2 = solve([f2])  # 第二个方程的解集，有2个解
        
        # 处理解
        if len(ans1) != 2:
            l = ans2[1][l]  # 设备距离最大值
        else:
            if ans1[1][l] <= ans2[1][l]:
                l = ans2[1][l]
            else:
                l = min(ans1[0][l], ans2[1][l])
        if output == "dist":
            return l
        # 满足约束的最优值
        M = acos((h * h + R * R - l * l) / 2 / R / h)  # ∠3的最大值
        if M >= (up_ang + down_ang) / 2:
            return 90 - up_ang / 2
        else:
            return 90 - down_ang / 2 - M

# 星地链路
def is_visible_or_dist(pos_sat, lon, lat, val, val_type="elevation_ang"):
    """
    通过地面站对卫星的仰角或距离，判断卫星是否可见

    Args: 
        pos_sat: 卫星xyz位置
        lon: 地面站经度
        lat: 地面站纬度
        val: 临界值，是最小仰角或最大距离
        val_type: 值类型，"elevation_ang"指临界值最小仰角，"dist"指最大距离

    Returns:
        若不可见，返回0
        若可见，返回星地距离
    """
    earth_r = 6371.393 #地球半径
    # 经纬度转为rad
    lat = radians(lat)
    lon = radians(lon)
    # 地面站xyz坐标
    x = earth_r * cos(lat) * cos(lon)
    y = earth_r * cos(lat) * sin(lon)
    z = earth_r * sin(lat)
    # print("地面站位置", x, y, z)
    # 矢量，地面站指向卫星
    dX = pos_sat[0] - x
    dY = pos_sat[1] - y
    dZ = pos_sat[2] - z
    # 星地距离
    dist = sqrt(dX**2 + dY**2 + dZ**2)
    # 根据指标判断可见性
    if val_type == "dist":
        return dist if val >= dist else 0
    else:
        # 将矢量转换为 ENU 坐标
        t = -sin(lon) * dX + cos(lon) * dY
        n = -sin(lat) * cos(lon) * dX - sin(lat) * sin(lon) * dY + cos(lat) * dZ
        u = cos(lat) * cos(lon) * dX + cos(lat) * sin(lon) * dY + sin(lat) * dZ
        # 仰角
        alt_zeta = degrees(atan2(u, sqrt(t**2 + n**2)))
        # print("地面站仰角：", alt_zeta, "度")
        # 和最小仰角进行比较，若比它还小，说明不可见
        return dist if alt_zeta >= val else 0



def get_sat_dist(sat1: SatelliteNode, sat2: SatelliteNode, time: Time):
    """
    计算卫星间的距离, 若不可见则返回inf
    
    Args: 
        sat1, sat2: SatelliteNode
    
    Returns:
        卫星间的角度，不可见则返回inf
    """
    earth_r = 6371.393 #地球半径
    # # 计算向量夹角，保证acos不出错
    # pos1 = sat1._satellite_pos(time)
    # pos2 = sat2._satellite_pos(time)
    # print(f"卫星1位置", pos1, f"卫星2位置", pos2)
    # # sat1,sat2的高度应该是相同的
    # cosL = (pos1[0]*pos2[0]+pos1[1]*pos2[1]+pos1[2]*pos2[2])/sat1.h/sat2.h
    # if cosL <= -1:
    #     L = pi
    # elif cosL >= 1:
    #     L = 0
    # else:
    #     L = acos(cosL)
    # # 若被地球挡住，则不可见；否则返回两星距离
    # if sat1.h * cos(L / 2) <= earth_r:
    #     return np.inf
    # else:
    #     return 2 * sat1.h * sin(L / 2)
    pos1 = np.array(sat1._satellite_pos(time))
    pos2 = np.array(sat2._satellite_pos(time))
    # print(f"卫星1位置", pos1, f"卫星2位置", pos2)
    # 欧氏距离
    distance = np.linalg.norm(pos1 - pos2)
    
    # 取两卫星连线中点
    midpoint = 0.5 * (pos1 + pos2)
    midpoint_norm = np.linalg.norm(midpoint)
    
    # 如果中点在地球半径以内，说明连线被地球遮挡
    if midpoint_norm < earth_r:
        return np.inf
    elif distance > 3000:
        return np.inf
    else:
        return distance

def link_data_rate(sat1: SatelliteNode, sat2: SatelliteNode, d, time: Time):
    """计算链路速率（根据公式(9)(10)）"""
    c = 3e8  # 光速
    k = 1.38e-23  # Boltzmann常数
    Un = 25  # 系统噪声温度（dBK）
    EbN0 = 1  # 接收能量/噪声谱密度
    A = 1.5  # 链路裕度（dB）
    carrier_freq = 23e9 # 载波频率

    if d == np.inf:
        return 0
    else:
        Lfs = (c / (4 * np.pi * d * carrier_freq)) ** 2
        R = (sat1.tran_power * sat1.tran_gain * sat2.rec_gain * Lfs) / (k * Un * EbN0 * A)
        return R

class Walker(object):
    """
    Walker星座类，用于创建卫星星座
    """
    def __init__(self, num_sats, h, angle, P_num, sat_comp_resource:list, 
                 sat_tran_power:list, sat_tran_gain:list, sat_rec_gain:list):
        self.num_sats = num_sats
        self.h = h
        self.angle = angle
        self.P_num = P_num
        self.sat_comp_resource = sat_comp_resource
        self.sat_tran_power = sat_tran_power
        self.sat_tran_gain = sat_tran_gain
        self.sat_rec_gain = sat_rec_gain

    def _generate_tles_line2(self, N, h, i, P, F=int(1)):
        """
        生成walker星座中所有卫星的TLE星历的第二行
        Args:
            N: int，walker星座中卫星总数
            h: float，卫星轨道高度，单位km
            i: float，卫星轨道倾角，单位度
            P: int，walker星座的轨道面数
            F: int，walker星座的相位数，默认为1
        Returns:
            list，包含所有卫星的第二行TLE
            STARLINK-1010
            1 44716U 19074D   25187.23278464  .00110409  00000+0  17717-2 0  9991
            2 44716  53.0543  195.3554  0010293 348.5237  11.5536 15.52174921312007
                    轨道倾角 升交点赤经 轨道偏心率 升交点角距 平近点角 每日平均运动
        """
        GM = 3986005 * 10 ** 8      # 【WGS-84】地球引力和地球质量的乘积
        def _tle_format(num, all=8, dec=4):
            return str(f"%.{dec}f"%num).zfill(all)
        
        tles = []                     # 计算各卫星tle，并加入该列表
        detu = 360 / N * F  # 邻轨对应卫星间的相位差
        # walker星座各卫星每天绕地圈数
        circles = sqrt(GM) * 12 * 3600 / pi / pow(h*1000, 1.5)
        num_S = int(N/P)  # 每个轨道面上的卫星数

        for sat_id in range(N):
            Pm = int(sat_id / num_S)  # 轨道面编号，0 ~ P-1
            Nm = sat_id % num_S       # 轨道内编号，0 ~ S-1
            omega_m = 90 / P * Pm          # 升交点赤经 omega_m = 180 / P * Pm
            # 测试生成附近的几颗卫星
            u_m = 0 + (60 / num_S * Nm) % 360 + detu * Pm  # 平近点角 u_m = 360 / num_S * Nm + detu * Pm
            tles.append(
                f'2 44716 {_tle_format(i)} {_tle_format(omega_m)} 0000000 000.0000 '
                f'{_tle_format(u_m)} {_tle_format(circles,11,8)}'
            )
        return tles
        # 如果需要以STARLINK-1010为基准生成相邻的卫星tle
        # base_mean_anomaly = 11.5536
        # num_sats = 5
        # mean_motion = 15.52174921
        # inclination = 53.0543
        # raan = 195.3554
        # ecc = 0.0010293
        # arg_perigee = 348.5237

        # tles = []
        # for i in range(num_sats):
        #     mean_anomaly = (base_mean_anomaly + i * (360/num_sats)) % 360
        #     tle_line2 = (
        #         f"2 44716 "
        #         f"{inclination:8.4f} "
        #         f"{raan:8.4f} "
        #         f"{ecc*1e7:07.0f} "
        #         f"{arg_perigee:8.4f} "
        #         f"{mean_anomaly:8.4f} "
        #         f"{mean_motion:11.8f} 99999"
        #     )
        #     tles.append(tle_line2)

    def create_satellites(self):
        """
        创建卫星星座并返回Satellite对象列表
        """
        tle_list_line2 = self._generate_tles_line2(self.num_sats, self.h, self.angle, self.P_num)
        satellite_list = []

        for i in range(self.num_sats):
            # 生成TLE数据
            tle_line1 = f"1 44716U 19074D   25187.23278464  .00110409  00000+0  17717-2 0  9991"
            tle_line2 = tle_list_line2[i]
            #print(f"Creating Satellite {i} with TLE:\n{tle_line1}\n{tle_line2}")

            # 创建Satellite对象
            sat = Satellite(
                id=i,
                h=self.h,
                i=self.angle,
                comp_resource=self.sat_comp_resource[i],
                tran_power=self.sat_tran_power[i],
                tran_gain=self.sat_tran_gain[i],
                rec_gain=self.sat_rec_gain[i],
                tle_line1=tle_line1,
                tle_line2=tle_line2
            )
            satellite_list.append(sat)


        return satellite_list
    
    def _update_sat_links(self, time: Time, satellites, sat_topology, sat_links):
        """
        Args:
        time: 当前时间
        satellites: 卫星列表:Satellite对象列表
        sat_topology: 卫星拓扑结构，字典形式
        sat_links: 卫星链路信息，字典形式
        """
        """
        采用grid结构：
        - 同轨相邻卫星连接
        - 邻轨相邻卫星连接
        更新self.sat_topology, self.sat_links
        """
        S = int(self.num_sats / self.P_num)  # 每个轨道面上的卫星数
        sat_topology.clear()
        sat_links.clear()

        for i, sat in enumerate(satellites):
            sat.target_sat_list.clear()
            sat_topology[sat.id] = []

        # 同轨相邻连接
        # print("[DEBUG] 开始同轨相邻连接...")
        for p in range(self.P_num):
            for s in range(S):
                a_idx = p * S + s
                b_idx = p * S + (s + 1) % S
                sat1 = satellites[a_idx]
                sat2 = satellites[b_idx]
                dist = get_sat_dist(sat1, sat2, time) #卫星必须满足可见关系
                if dist == np.inf:
                    continue
                rate = link_data_rate(sat1, sat2, dist, time)
                #print(f"[DEBUG] 同轨连接: sat{sat1.id}(轨道{p},位置{s},索引{a_idx}) <-> sat{sat2.id}(轨道{p},位置{(s+1)%S},索引{b_idx})")
                self._add_link(sat1, sat2, dist, rate, sat_topology, sat_links)

        # 邻轨相邻连接
        # print("[DEBUG] 开始邻轨相邻连接...")
        if self.P_num > 1:
            for p in range(self.P_num):
                for s in range(S):
                    a_idx = p * S + s
                    b_p = (p + 1) % self.P_num
                    b_idx = b_p * S + s
                    sat1 = satellites[a_idx]
                    sat2 = satellites[b_idx]
                    dist = get_sat_dist(sat1, sat2, time) #卫星必须满足可见关系
                    if dist == np.inf:
                        continue
                    rate = link_data_rate(sat1, sat2, dist, time)
                    #print(f"[DEBUG] 邻轨连接: sat{sat1.id}(轨道{p},位置{s},索引{a_idx}) <-> sat{sat2.id}(轨道{b_p},位置{s},索引{b_idx})")
                    self._add_link(sat1, sat2, dist, rate, sat_topology, sat_links)

        print("卫星间grid连接关系", sat_topology)
        print("卫星间grid链路信息", sat_links)
    
    def _add_link(self, sat1, sat2, dist, rate, sat_topology, sat_links):
        """
        将sat1和sat2的双向连接加入拓扑
        """
        # print(f"[DEBUG] _add_link 被调用: sat{sat1.id} <-> sat{sat2.id}, 距离={dist:.2f}, 速率={rate:.2e}")
        
        # 避免自己连接到自己
        if sat1.id == sat2.id:
            print(f"[警告] 尝试连接卫星到自己: sat{sat1.id}")
            return
            
        # 避免重复连接
        if sat2.id in sat_topology[sat1.id]:
            print(f"[警告] 重复连接: sat{sat1.id} -> sat{sat2.id}")
            print(f"[DEBUG] sat{sat1.id} 当前连接列表: {sat_topology[sat1.id]}")
            return
            
        if sat1.id in sat_topology[sat2.id]:
            print(f"[警告] 重复连接: sat{sat2.id} -> sat{sat1.id}")
            print(f"[DEBUG] sat{sat2.id} 当前连接列表: {sat_topology[sat2.id]}")
            return
        
        # 添加连接
        sat_topology[sat1.id].append(sat2.id)
        sat_links[(sat1.id, sat2.id)] = {
            "distance": dist,
            "data_rate": rate
        }
        sat1.target_sat_list.append(sat2)

        sat_topology[sat2.id].append(sat1.id)
        sat_links[(sat2.id, sat1.id)] = {
            "distance": dist,
            "data_rate": rate
        }
        sat2.target_sat_list.append(sat1)
    
    

# 卫星世界类
class SatelliteWorld(object):
    """
    参考mpe环境的World类
    """
    def __init__(self, walker:Walker, time: Time):
        '''环境相关属性'''
        # walker星座对象
        self.walker = walker
        # 卫星列表，元素是Satellite对象
        self.satellites = []
        # 用户簇列表，元素是UserCluster对象
        self.user_clusters = []
        # 最大时间步——在创建world时用脚本中的episode_length赋值
        self.world_length = 100
        # 当前时间步
        self.world_step = 0
        # 智能体数量
        self.num_agents = 0
        # 物理世界的时间，对应年月日
        # self.time = 0  # 删除float类型
        # 时间步长，也就是world_step一步对应的物理世界的时间
        self.dt = 1.0
        # 新增：当前物理世界的时间对象
        self.current_time = time
        self.initial_time = time

        # # 兼容MPE环境的属性
        # self.dim_c = 0  # 通信维度，卫星环境暂时不需要
        # self.dim_p = 3  # 位置维度，卫星环境为3D

        '''拓扑更新'''
        # 邻接表结构，表示卫星与其他卫星的连接关系，例如 {0: [1, 2], 1: [0, 2, 3], ...}
        self.sat_topology = {}
        # 具体链路信息 ,例如{(i, j): {"distance": ..., "data_rate": ..., "visible_time": ...}}
        self.sat_links = {}
        # 用户-卫星链路信息，例如 {(user_id, sat_id): 距离，若不可见为-1}
        self.user_sat_visibility = {}

        # 奖励权重
        self.delay_weight = 1.0
        self.migration_cost_weight = 1.0
        # # 链路参数
        # self.link_params = {
        #     'min_bandwidth': 1.0,    # 最小带宽
        #     'max_bandwidth': 10.0    # 最大带宽
        # }

    def step(self):
        """
        环境步进函数，是环境的核心函数，负责模拟整个物理世界的一个时间步的演进。
        """
        # 更新当前时间步
        self.world_step += 1
        # 更新物理世界的时间（自增秒数）
        self.current_time.second += self.dt
        # 处理进位
        if self.current_time.second >= 60:
            self.current_time.minute += int(self.current_time.second // 60)
            self.current_time.second = self.current_time.second % 60
        if self.current_time.minute >= 60:
            self.current_time.hour += int(self.current_time.minute // 60)
            self.current_time.minute = self.current_time.minute % 60
        if self.current_time.hour >= 24:
            self.current_time.day += int(self.current_time.hour // 24)
            self.current_time.hour = self.current_time.hour % 24
        # TODO: 可进一步处理月份和年份进位
        print("当前物理世界时间:", self.current_time)
        #1. 获取所有卫星的动作，由外部策略网络传入，这里不需要管
        #由environment.py的step函数获取动作

        #2. 执行任务迁移动作
        for agent in self.satellites:
            self._perform_task_migration(agent)
        
        #3. 更新卫星间的链路状态，包括距离、可见时间和链路速率
        # self._update_link_states(self.current_time)
        self._update_link_states(self.current_time)

        #4. 更新可见性信息--用户和卫星
        self._update_visibility_matrix(self.current_time)
        
        # #5. 计算延迟
        # total_delay = self._calculate_total_delay()

        # #6. 计算服务迁移成本
        # migration_cost = self._calculate_migration_cost()
        
        # #7. 计算奖励
        # rewards = self._calculate_rewards(total_delay, migration_cost)
    
    def _action_explain(self, agent: Satellite):
        """
        将动作编码转换为返回的对象
        Args:
            agent: Satellite对象，当前执行迁移的卫星
        Returns:
            target_sat: Satellite对象，表示迁移的目标卫星
            service_instance: ServiceInstance对象，表示迁移的服务实例
            user: UserCluster对象，表示请求该服务的用户
        """
        # 解析迁移动作
        service_id, target_id = agent.action.decode(agent.target_sat_list)
        if service_id == -1:
            return  # 不迁移
        if service_id is None or target_id is None:
            return
        
        # 通过target_id在目标卫星列表中查找对应的卫星对象
        # target_sat : Satellite对象
        target_sat = None
        for sat in agent.target_sat_list:
            if sat.id == target_id:
                target_sat = sat
                break
        if target_sat is None:
            print(f"[错误] 目标卫星 {target_id} 不在 agent {agent.id} 的 target_sat_list 中，跳过迁移。")
            return
        print("[DEBUG] 迁移目标卫星为", target_sat, target_sat.id)

        # 获取迁移的服务实例和用户
        # 在卫星的instance_list中查找对应service_id的ServiceInstance：Instance对象
        service_instance = None
        for instance in agent.instance_list:
            if instance.service_id == service_id:
                service_instance = instance
                break
        if service_instance is None:
            print(f"[错误] service_instance id={service_id} 不在 agent {agent.id} 的 instance_list 中，跳过迁移。")
            return
        print(f"[DEBUG] 待迁移的服务为: {service_instance.service_id}")

        # 通过service_instance找到请求该服务的用户
        user = None
        for user_cluster in self.user_clusters:
            if user_cluster.service_instance.service_id == service_id:
                user = user_cluster
                break
        if user is None:
            return
        
        print(f"[DEBUG] _action_explain成功返回: target_sat={target_sat.id}, service_instance={service_instance.service_id}, user={user.id}")
        return target_sat, service_instance, user
        

    
    def _perform_task_migration(self, agent: Satellite):
        """
        执行任务迁移
        Args:
            agent: Satellite对象，当前执行迁移的卫星
        """
        # 1. 对迁移动作进行处理
        result = self._action_explain(agent)
        if result is None or len(result) != 3:
            return  # 动作不合法或返回值不完整，跳过迁移
        
        target_sat, service_instance, user = result
        print("[debug] 执行迁移 agent {}: service_id={}, target_id={}".format(agent.id, service_instance.service_id, target_sat.id))
        
        # 更新用户的当前卫星
        user.current_sat = target_sat
        print("[debug] 请求该服务的用户为", user, user.id)

        # 3. 检查目标卫星是否有足够的计算资源
        if target_sat.comp_resource < service_instance.instance_size:
            return

        # 4. 执行迁移，更新资源、源节点和目标节点的instance_list和service_users
        # 4.1 更新源卫星资源
        agent.comp_resource += service_instance.instance_size
        agent.instance_list.remove(service_instance)
        agent.service_users.remove(user)
        print("[debug] 迁移后源卫星实例列表",agent.instance_list)
        # 4.2 更新目标卫星资源
        target_sat.comp_resource -= service_instance.instance_size
        target_sat.instance_list.append(service_instance)
        target_sat.service_users.append(user)
        print("[debug] 迁移后目标卫星实例列表",target_sat.instance_list)
        # 4.3 更新用户的当前卫星
        user.current_sat = target_sat

        # # 5. 重置动作
        # agent.action = SatelliteAction()

    
    def _update_link_states(self, time: Time):
        """
        封装
        更新全局的卫星间链路状态，包括距离、可见时间和链路速率等信息。
        更新self.sat_topology和self.sat_links
        更新Satellite类的target_sat属性
        """
        
        # 清空现有拓扑和链路信息，确保每次调用都是全新的
        self.sat_topology.clear()
        self.sat_links.clear()
        for sat in self.satellites:
            sat.target_sat_list.clear()
            
        self.walker._update_sat_links(self.current_time, self.satellites, self.sat_topology, self.sat_links)
    #     self.sat_topology.clear()
    #     self.sat_links.clear()
    #     # 遍历world中所有卫星，计算与其他卫星的距离和链路速率
    #     for i, sat1 in enumerate(self.satellites):
    #         sat1.target_sat_list.clear()
    #         self.sat_topology[sat1.id] = []
    #         for j, sat2 in enumerate(self.satellites):
    #             if i == j:
    #                 continue
    #             dist = get_sat_dist(sat1, sat2, time)
    #             # 打印调试信息
    #             # print("卫星间距离 sat{} and sat{}: {}".format(sat1.id, sat2.id, dist))
    #             if dist == np.inf:
    #                 continue
    #             rate = link_data_rate(sat1, sat2, dist, time)
    #             # 更新全局卫星拓扑和链路信息
    #             self.sat_topology[sat1.id].append(sat2.id)
    #             self.sat_links[(sat1.id, sat2.id)] = {
    #                 "distance": dist,
    #                 "data_rate": rate
    #             }
    #             # 更新卫星的target_sat_list
    #             sat1.target_sat_list.append(sat2)
    #     print("卫星间连接关系", self.sat_topology)
    
    def _update_visibility_matrix(self, time: Time):
        '''
        更新用户和卫星的可见性
        更新self.user_sat_visibility和sat.visible_user
        '''
        self.user_sat_visibility.clear()
        for sat in self.satellites:
            sat.visible_user.clear()
            for user in self.user_clusters:
                dist = sat._get_visible_user(user, time)
                # 打印调试信息
                # print("计算可见性 user{} and sat{}: {}".format(user.id, sat.id, dist))
                # 更新全局的用户-卫星链路信息-可见性和距离，不可见则dist=-1
                self.user_sat_visibility[(user.id, sat.id)] = dist
                # 如果可见，则更新卫星的visible_user列表中
                if dist:
                    sat.visible_user.append(user)
        print("用户-卫星可见性信息", self.user_sat_visibility)

    def _calculate_total_delay(self, sat:Satellite):
        """
        计算卫星sat迁移服务的总延迟
        """
        print(f"[DEBUG] _calculate_total_delay被调用，卫星{sat.id}")
        
        total_delay = 0.0
        compute_delay = 0.0
        comm_delay = 0.0
        
        result = self._action_explain(sat)
        if result is None or len(result) != 3:
            migration_delay = 0.0
        else: 
            target_sat, service_instance, user = result
            # 迁移延迟, 卫星每次只能迁移一个服务实例
            migration_delay = self._compute_migration_delay(sat, user, target_sat)

        # 对卫星服务的所有用户计算通信+计算延迟
        for user in sat.service_users:
            # 计算延迟
            compute_delay += self._compute_computation_delay(user, sat)
            # 通信延迟
            comm_delay += self._compute_communication_delay(user, sat)
        
        total_delay += migration_delay + compute_delay + comm_delay
        print(f"迁移延迟: {migration_delay}, 计算延迟: {compute_delay}, 通信延迟: {comm_delay}")
        return total_delay
    

    def _compute_migration_delay(self, sat:Satellite, user:UserCluster, target_sat:Satellite):
        """
        TODO：服务迁移延迟 = （实例传输延迟 + 传播延迟） + 服务停止时间 + 服务启动时间
        """
        delay = 0.0
        # 光速
        C = 3e8  # m/s
        ins_size = user.service_instance.instance_size
        rate = self.sat_links[(sat.id, target_sat.id)]["data_rate"]
        dist = self.sat_links[(sat.id, target_sat.id)]["distance"]
        delay = ins_size / rate + dist / C
        return delay


    def _compute_computation_delay(self, user: UserCluster, sat: Satellite, eta=10):
        """
        计算延迟
        :param task_size: float, 任务大小 (Mbit)
        :param cpu_available: float, 可用 CPU 资源 (Gcycles/s)
        :param eta: float, 每 Mbit 所需 CPU cycles (默认 10 cycles/Mbit)
        :return: float, computation delay (s)
        """
        # 假设卫星资源被用户平分 TODO: 判断这里是否合理以及与资源更新的计算先后
        cpu_available = sat.comp_resource / len(sat.service_users)
        return (eta * user.task_size) / (cpu_available * 1e3)  # 注意 Gcycles → Mcycles


    def _compute_communication_delay(self, user: UserCluster, sat: Satellite):
        """
        计算用户user到卫星sat的通信延迟
        """
        W_task = user.task_size  # Mbit
         # 获取链路信息
        if user.current_sat is None:
            print(f"[错误] 用户{user.id}的current_sat为None，无法计算通信延迟")
            return float('inf')
        # 获取链路信息
        d_us = self.user_sat_visibility.get((user.current_sat.id, sat.id))
        if d_us == -1:
            print(f"[错误] 用户{user.id}到卫星{sat.id}的距离为-1")
            return float('inf')

        # 2. 获取参数
        D_u = user.task_size  # 上传数据量，单位Mbit
        c = 3e8  # 光速

        # 3. 信道参数
        B_ui = 10  # 单位MHz，默认10MHz
        P_u = 1     # 单位W，默认1W
        h_ui = 1e-4  # 默认1e-4
        N0 = 1e-9   # 单位W/Hz，默认1e-9

        # 4. Shannon容量公式
        # 注意带宽单位需统一，假设D_u单位为Mbit，B_ui单位为MHz，需转为bit/s和Hz
        B_ui_Hz = B_ui * 1e6
        R_ui = B_ui_Hz * log2(1 + (P_u * h_ui) / (N0 * B_ui_Hz))  # 单位bit/s

        if R_ui == 0:
            return float('inf')

        # 5. 计算延迟
        D_u_bit = D_u * 1e6  # Mbit -> bit
        comm_delay = D_u_bit / R_ui + d_us / c  # 单位：秒

        return comm_delay



    def _calculate_rewards(self, load_imbalance, total_delay):
        """
        计算奖励
        """
        rewards = []
        for satellite in self.satellites:
            reward = -(self.load_balance_weight * load_imbalance + 
                      self.delay_weight * total_delay)
            rewards.append(reward)
        return rewards


    def _calculate_visible_time(self, sat1, sat2):
        """
        计算两颗卫星的可见时间
        """
        # 实现可见时间计算逻辑
        pass

    
    def plot_step_positions(self, step_idx, save_dir="satellite_steps"):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 1. 绘制地球球体
        r = 6371  # 地球半径，单位km
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        ax.plot_surface(x, y, z, color='deepskyblue', alpha=0.3)

        # 2. 绘制所有卫星
        sat_x, sat_y, sat_z = [], [], []
        for sat in self.satellites:
            pos = sat._satellite_pos(self.current_time)  # 获取当前step卫星位置
            sat_x.append(pos[0])
            sat_y.append(pos[1])
            sat_z.append(pos[2])
            ax.text(pos[0], pos[1], pos[2], f"S{sat.id}", fontsize=8, color='red')  # 标注卫星编号
        ax.scatter(sat_x, sat_y, sat_z, c='red', marker='o', label='Satellites')

        # 3. 绘制所有用户
        user_x, user_y, user_z = [], [], []
        for user in self.user_clusters:
            # 需实现经纬度到xyz的转换
            lon, lat = user.lon, user.lat
            pos = [
                r * np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                r * np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                r * np.sin(np.radians(lat))
            ]
            user_x.append(pos[0])
            user_y.append(pos[1])
            user_z.append(pos[2])
            ax.text(pos[0], pos[1], pos[2], f"U{user.id}", fontsize=8, color='green')  # 标注用户编号
        ax.scatter(user_x, user_y, user_z, c='green', marker='^', label='Users')

        # 设置视角
        ax.view_init(elev=30, azim=60)

        ax.set_title(f"Step {step_idx} 卫星与用户三维分布")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        ax.legend()

        plt.savefig(os.path.join(save_dir, f"satellite_step_{step_idx}.png"))
        plt.close()

    

    def plot_step_positions_interactive(self, step_idx, save_dir="satellite_steps_html"):
        """
        使用 Plotly 绘制卫星与用户三维分布，并保存为可交互 HTML 图像。

        Args:
            step_idx: 当前时间步编号
            save_dir: HTML 文件保存路径
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        r = 6371  # 地球半径

        # ---------------- 地球球体 ---------------- #
        u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)

        surface = go.Surface(
            x=x, y=y, z=z,
            colorscale='Blues',
            opacity=0.5,
            showscale=False,
            name="Earth"
        )

        # ---------------- 卫星 ---------------- #
        sat_x, sat_y, sat_z, sat_labels = [], [], [], []
        sat_positions = {}  # 存储卫星位置，用于绘制连接线
        for idx, sat in enumerate(self.satellites):
            pos = sat._satellite_pos(self.current_time)
            sat_x.append(pos[0])
            sat_y.append(pos[1])
            sat_z.append(pos[2])
            sat_labels.append(f"S{idx}")
            sat_positions[sat.id] = pos

        sat_trace = go.Scatter3d(
            x=sat_x, y=sat_y, z=sat_z,
            mode='markers+text',
            marker=dict(size=4, color='red'),
            text=sat_labels,
            textposition="top center",
            name="Satellites"
        )

        # ---------------- 用户 ---------------- #
        user_x, user_y, user_z, user_labels = [], [], [], []
        user_positions = {}  # 存储用户位置，用于绘制连接线
        for idx, user in enumerate(self.user_clusters):
            lon, lat = user.lon, user.lat
            pos = [
                r * np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                r * np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                r * np.sin(np.radians(lat))
            ]
            user_x.append(pos[0])
            user_y.append(pos[1])
            user_z.append(pos[2])
            user_labels.append(f"U{idx}")
            user_positions[user.id] = pos

        user_trace = go.Scatter3d(
            x=user_x, y=user_y, z=user_z,
            mode='markers+text',
            marker=dict(size=5, color='green', symbol='diamond'),
            text=user_labels,
            textposition="top center",
            name="Users"
        )

        # ---------------- 卫星间连接线 ---------------- #
        sat_links_traces = []
        for sat1_id, connected_sats in self.sat_topology.items():
            if sat1_id in sat_positions:
                sat1_pos = sat_positions[sat1_id]
                for sat2_id in connected_sats:
                    if sat2_id in sat_positions and sat1_id < sat2_id:  # 避免重复绘制
                        sat2_pos = sat_positions[sat2_id]
                        
                            
                        link_trace = go.Scatter3d(
                            x=[sat1_pos[0], sat2_pos[0]],
                            y=[sat1_pos[1], sat2_pos[1]],
                            z=[sat1_pos[2], sat2_pos[2]],
                            mode='lines',
                            line=dict(color='blue', width=2, dash='dash'),
                            showlegend=False,
                            name=f"Sat-Sat Link"
                        )
                        sat_links_traces.append(link_trace)

        # ---------------- 用户-卫星连接线 ---------------- #
        user_sat_links_traces = []
        for user in self.user_clusters:
            if user.current_sat is not None and user.id in user_positions and user.current_sat.id in sat_positions:
                user_pos = user_positions[user.id]
                sat_pos = sat_positions[user.current_sat.id]
                link_trace = go.Scatter3d(
                    x=[user_pos[0], sat_pos[0]],
                    y=[user_pos[1], sat_pos[1]],
                    z=[user_pos[2], sat_pos[2]],
                    mode='lines',
                    line=dict(color='orange', width=3, dash='dot'),
                    showlegend=False,
                    name=f"User-Sat Link"
                )
                user_sat_links_traces.append(link_trace)

        # ---------------- 可见性连接线（虚线） ---------------- #
        visibility_links_traces = []
        for user in self.user_clusters:
            if user.id in user_positions:
                user_pos = user_positions[user.id]
                for sat in self.satellites:
                    if sat.id in sat_positions:
                        # 检查可见性
                        visibility_key = (user.id, sat.id)
                        if visibility_key in self.user_sat_visibility and self.user_sat_visibility[visibility_key] > 0:
                            sat_pos = sat_positions[sat.id]
                            # 只绘制非当前服务卫星的可见性连接
                            if user.current_sat is None or user.current_sat.id != sat.id:
                                link_trace = go.Scatter3d(
                                    x=[user_pos[0], sat_pos[0]],
                                    y=[user_pos[1], sat_pos[1]],
                                    z=[user_pos[2], sat_pos[2]],
                                    mode='lines',
                                    line=dict(color='gray', width=1, dash='dash'),
                                    opacity=0.3,
                                    showlegend=False,
                                    name=f"Visibility Link"
                                )
                                visibility_links_traces.append(link_trace)

        # ---------------- 布局 ---------------- #
        layout = go.Layout(
            title=f"Step {step_idx} 卫星与用户三维分布",
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)",
                aspectmode='data'
            ),
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # 合并所有轨迹
        all_traces = [surface, sat_trace, user_trace] + sat_links_traces + user_sat_links_traces + visibility_links_traces
        fig = go.Figure(data=all_traces, layout=layout)

        # ---------------- 保存 HTML ---------------- #
        save_path = os.path.join(save_dir, f"satellite_step_{step_idx}.html")
        fig.write_html(save_path)

        print(f"已保存交互式三维图: {save_path}")
        print(f"连接信息: 卫星间连接 {len(sat_links_traces)} 条, 用户-卫星服务连接 {len(user_sat_links_traces)} 条, 可见性连接 {len(visibility_links_traces)} 条")
