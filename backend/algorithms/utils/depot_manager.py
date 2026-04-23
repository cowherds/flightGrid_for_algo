"""
站点管理模块

本模块提供无人机站点（仓库/配送中心）的时间槽位管理功能。
用于处理多架无人机在同一站点的起飞和降落调度，避免时间冲突。

主要功能：
- 管理站点的占用时间槽位
- 计算无人机可用的起飞/降落时间
- 处理多架无人机的时间冲突

使用场景：
- 调度算法中，为无人机分配站点的起飞和降落时间
- 确保多架无人机不会在同一时间使用同一站点
"""


class DepotManager:
    """
    站点管理器

    用于管理无人机站点的时间槽位占用情况。每个站点可以同时处理多架无人机，
    但需要确保它们的起飞和降落时间不重叠。

    属性：
        occupied_slots (Dict[str, List[Tuple[float, float]]]):
            站点的占用时间槽位字典。
            键为站点 ID，值为 [(开始时间, 结束时间), ...] 的列表
        occupy_duration (float):
            每次占用站点的持续时间，单位秒。默认为 5 秒。
            表示一架无人机从开始起飞到完全离开站点所需的时间。

    示例：
        >>> manager = DepotManager(occupy_duration=5.0)
        >>> # 查询站点 'depot_1' 在时间 100s 的可用时间
        >>> available_time = manager.get_available_time('depot_1', 100.0)
        >>> print(f"可用时间: {available_time}s")
    """

    def __init__(self, occupy_duration: float = 5.0):
        """
        初始化站点管理器

        创建一个新的站点管理器实例，用于管理多个站点的时间槽位占用情况。
        每个站点可以同时处理多架无人机，但需要确保它们的起飞和降落时间不重叠。

        Args:
            occupy_duration (float): 每次占用站点的持续时间，单位秒。
                默认为 5 秒。表示一架无人机从开始起飞到完全离开站点所需的时间。
                这个值应该根据实际的无人机起降时间进行调整。

        属性初始化：
            - occupied_slots: 空字典，用于存储每个站点的占用时间槽位
            - occupy_duration: 保存占用时间参数

        示例：
            >>> manager = DepotManager(occupy_duration=5.0)
            >>> # 现在可以使用 manager 来管理站点的时间槽位
        """
        # 存储每个站点的占用时间槽位
        # 数据结构：{站点ID: [(开始时间, 结束时间), ...]}
        self.occupied_slots = {}
        # 每次占用的持续时间，单位秒
        self.occupy_duration = occupy_duration

    def get_available_time(self, depot_id: str, request_time: float) -> float:
        """
        获取站点的可用时间

        根据请求时间和站点的当前占用情况，计算无人机可以使用该站点的实际时间。
        如果请求时间已被占用，则返回下一个可用的时间。这个方法确保多架无人机
        不会在同一时间使用同一站点。

        算法流程：
        1. 如果站点不存在，初始化其占用槽位列表
        2. 对所有占用槽位按开始时间排序
        3. 遍历占用槽位，如果请求时间落在某个槽位内，则更新请求时间为该槽位的结束时间
        4. 在计算出的可用时间添加新的占用槽位
        5. 返回可用时间

        Args:
            depot_id (str): 站点 ID，用于唯一标识一个站点
            request_time (float): 请求的时间，单位秒

        Returns:
            float: 实际可用的时间，单位秒。
                如果请求时间未被占用，返回请求时间；
                如果被占用，返回下一个可用的时间。

        时间复杂度：O(n log n) 其中n为该站点的占用槽位数量
        空间复杂度：O(n)

        调用场景：
            - 调度算法为无人机分配起飞时间时调用
            - 确保多架无人机不会在同一时间使用同一站点
            - 生成无人机的起飞时间表

        示例：
            >>> manager = DepotManager(occupy_duration=5.0)
            >>> # 第一架无人机在时间 100s 请求站点
            >>> time1 = manager.get_available_time('depot_1', 100.0)
            >>> print(f"无人机1可用时间: {time1}s")  # 输出: 100.0
            >>> # 第二架无人机在时间 102s 请求同一站点
            >>> time2 = manager.get_available_time('depot_1', 102.0)
            >>> print(f"无人机2可用时间: {time2}s")  # 输出: 105.0（因为第一架占用到 105s）
            >>> # 第三架无人机在时间 110s 请求同一站点
            >>> time3 = manager.get_available_time('depot_1', 110.0)
            >>> print(f"无人机3可用时间: {time3}s")  # 输出: 110.0（因为 110s 已经可用）
        """
        # 如果站点不存在，初始化其占用槽位列表
        if depot_id not in self.occupied_slots:
            self.occupied_slots[depot_id] = []

        # 获取该站点的所有占用槽位，并按开始时间排序
        # 排序确保我们能够按时间顺序检查冲突
        slots = sorted(self.occupied_slots[depot_id])
        # 当前请求的可用时间，初始值为请求时间
        current_req = request_time

        # 遍历所有占用槽位，检查是否与请求时间冲突
        for (st, et) in slots:
            # 如果请求时间落在某个占用槽位内 [st, et)
            # 这意味着该时间已被其他无人机占用
            if st <= current_req < et:
                # 将请求时间更新为该槽位的结束时间
                # 这样可以找到下一个可用的时间
                current_req = et

        # 在计算出的可用时间添加新的占用槽位
        # 占用时间段为 [current_req, current_req + occupy_duration)
        # 这表示无人机将在 current_req 到 current_req + occupy_duration 期间占用该站点
        self.occupied_slots[depot_id].append((current_req, current_req + self.occupy_duration))

        # 返回实际可用的时间
        return current_req